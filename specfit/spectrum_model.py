from __future__ import division

from warnings import warn

import emcee
import lineid_plot
import numpy as np
from collections import defaultdict, OrderedDict

from corner import corner
from scipy.signal import medfilt
from scipy.stats import norm, truncnorm
from theano import tensor as T
from theano import shared
from theano import function as tfunction
from theano.gof.graph import ancestors
from theano.ifelse import ifelse
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd

__all__ = ['SpectrumModel']

ninf = T.constant(-np.inf, dtype=np.float64)

def actual_truncnorm(a, b, pos, std):
    a, b = (a - pos) / std, (b - pos) / std
    return truncnorm(a, b, pos, std)

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def np_gaussian_emission_line(x, *p):
    """p = [centre, std, amp]"""
    return p[2] * np.exp(-1. * ((x - p[0])**2.) / (2*(p[1]**2.)))

def line_flux(amplitude, std):
    return amplitude * np.sqrt(2 * np.pi) * std

def characterise_peak(wavelengths, continuum_subtracted_flux, continuum_error, expected_peak_wavelength):
    """returns basic fit [lambda, std, amplitude] for a expected peak. Returns a stupidly small peak if not found"""
    regions = contiguous_regions(continuum_subtracted_flux > continuum_error)
    regions = regions[(regions[:, 1] - regions[:, 0]) > 1]  # extended regions
    region_wvls = wavelengths[regions]
    peak_region = region_wvls[
        (region_wvls[:, 1] >= expected_peak_wavelength) & (region_wvls[:, 0] <= expected_peak_wavelength)]
    if len(peak_region) == 0:
        warn("peak finding failed for {}, assuming no peak".format(expected_peak_wavelength))
        return [expected_peak_wavelength, 0.001, 0.]
    peak_region = peak_region[0]
    filt = (wavelengths >= peak_region[0]) & (wavelengths <= peak_region[1])

    from scipy.optimize import curve_fit
    guess = [expected_peak_wavelength, 0.5 * (peak_region[1] - peak_region[0]), 1.]

    popt, pcov = curve_fit(np_gaussian_emission_line, wavelengths[filt], continuum_subtracted_flux[filt], guess, bounds=(0, np.inf))
    if (popt[0] >= 0 and popt[1] >= 0 and popt[2] >= 0):
        return popt
    warn("peaking finding returned bad values ({}) for {}, assuming no peak for sanity".format(popt, expected_peak_wavelength))
    return [expected_peak_wavelength, 0.001, 0.]

def characterise_peaks(wavelengths, continuum_subtracted_flux, continuum_error, expected_peak_wavelengths):
    l = []
    for w in expected_peak_wavelengths:
        popt = characterise_peak(wavelengths, continuum_subtracted_flux, continuum_error, w)
        l.append(popt)
    return np.asarray(l)

def smooth_spectrum(flux, wavelength, wvl_window=200, n_window=None):
    if wvl_window is not None:
        n = (wavelength[-1] - wavelength[0]) / len(wavelength)  # ang / block
        window = (int(wvl_window / n / 2) * 2) + 1  # always have an odd window length
    else:
        window = n_window
    assert window < len(flux), "smoothing windows must be less than half the length of the spectrum"
    return medfilt(flux, window)

def gaussian_emission_line(x, centre, amplitude, width, shift):
    return amplitude * T.exp(-1. * ((x - centre + shift)**2.) / (2*width*width))

def gaussian_absorption_line(x, centre, amplitude, width, shift):
    return -gaussian_emission_line(x, centre, amplitude, width, shift)

def lnlike(y, err, model):
    residuals = ((model - y) / err)**2.
    return -0.5 * T.sum(residuals)

def get_parameter_dependencies(parameter):
    return [i for i in ancestors([parameter]) if i.name is not None and i is not parameter]


class Indexer(object):
    def __init__(self, feature):
        self.feature = feature

    def __repr__(self):
        return '<Indexer for {}: {}>' .format(self.feature.name, ','.join(self.feature.parameter_names))

    def get_param_value(self, param):
        print param
        for f in self.feature.parent_region.model.features:
            try:
                return f.fitted[f.parameters.keys()[f.parameters.values().index(param)]]
            except (KeyError, ValueError):
                pass

    def __getattr__(self, item):
        try:
            return self.feature.parent_region.model.sampler.chain[self.feature._chain_index[item]]
        except KeyError:
            try:
                r = self.feature.parameters[item]
                dependencies = [(self.feature.parent_region.model._theano_vars.index(i), i) for i in get_parameter_dependencies(r)]
                args = {self.feature.parent_region.model.parameter_names[d[0]]: self.get_param_value(d[1]) for d in dependencies}
                return r.eval(args)
            except (KeyError, ValueError):
                func, args = self.feature.derived_quantities[item]
                return func(*[self.__getattr__(i) for i in args])

    def __getitem__(self, item):
        return self.__getattr__(item)


class SpectralFeature(object):
    default_conditions = {}
    derived_quantities = []

    def __init__(self, name, function, theano_variables, parent_region=None):
        super(SpectralFeature, self).__init__()
        self.name = name
        self._function = function
        self.parameter_names = function.__code__.co_varnames
        assert self.parameter_names[0] == 'x', "x must be the first argument for a feature function"
        self.parameter_names = self.parameter_names[1:]
        assert len(self.parameter_names) == len(theano_variables), "Theano parameters misspecified"
        self.parameters = OrderedDict([(pname, p) for pname, p in zip(self.parameter_names, theano_variables)])
        self._parent_region = parent_region
        self.parameter_guesses = OrderedDict([])
        self._chain_index = {}

    @property
    def fitted(self):
        return Indexer(self)

    @property
    def parent_region(self):
        return self._parent_region

    def assign_to_region(self, region):
        self._parent_region = region

    def __getattr__(self, item):
        return self.parameters[item]

    def __getitem__(self, item):
        return self.parameters[item]

    def function(self, wavelength):
        return self._function(wavelength, *self.parameters.values())

    def __repr__(self):
        if self.parent_region is not None:
            parent_repr = 'in region {}-{}'.format(self.parent_region.min, self.parent_region.max)
        else:
            parent_repr = 'unaffiliated feature'
        s = "{} {} ({})\n".format(self.__class__.__name__, self.name, parent_repr)
        sep = '='*len(s) + "\n"
        s += sep
        for p, tp in zip(self.parameter_names, self.parameters.items()):
            if isinstance(tp[1], T.sharedvar.SharedVariable):
                s += '{} (fixed @ {})\n'.format(p, tp[1].get_value())
            else:
                deps = get_parameter_dependencies(tp[1])
                deps = [i.name for i in deps]
                if not deps:
                    s += '{} ({})\n'.format(p, tp[1].name)
                else:
                    s += '{} (depends on {})\n'.format(p, ', '.join(deps))
        return sep+s

    def _default_guess(self, w, f, e, smoothing_window):
        return truncnorm(0, np.inf, 0, 1)

    def _get_param_guess(self, par):
        try:
            return getattr(self, 'guess_' + par)
        except AttributeError:
            warn("{} doesn't have guess_{} as an attribute, guessing above 0")
            return self._default_guess()

    def guess(self, wavelength, flux, error, smoothing_window):
        """
        Guesses all parameters with default arguments
        """
        filter = self.parent_region.filter(wavelength)
        w, f, e = wavelength[filter], flux[filter], error[filter]
        for par in self.parameter_names:
            if not self.parameters[par] in self.parent_region.model._theano_shared:
                self.parameter_guesses[par] = self._get_param_guess(par)(w, f, e, smoothing_window)

    def add_manual_guess(self, **guess):
        for k, v in guess.iteritems():
            self.parameter_guesses[k] = v


class EmissionLine(SpectralFeature):
    default_conditions = {'amplitude': partial(T.lt, 0), 'width': partial(T.lt, 0)}
    derived_quantities = {'flux': [line_flux, ('amplitude', 'width')]}

    def __init__(self, name, centre=None, amplitude=None, width=None, shift=0, parent_region=None):
        super(EmissionLine, self).__init__(name, gaussian_emission_line, [centre, amplitude, width, shift], parent_region)

    def guess(self, wavelength, flux, error, smoothing_window):
        median_smoothed = smooth_spectrum(flux, wavelength, smoothing_window)
        filter = self.parent_region.filter(wavelength)
        w, smoothf, e = wavelength[filter], median_smoothed[filter], error[filter]
        cent, std, amp = characterise_peak(w, flux[filter]-smoothf, np.std(smoothf)*2, self.parameters['centre'].get_value())
        self.parameter_guesses['width'] = truncnorm(0, np.inf, std, std*0.1)
        self.parameter_guesses['amplitude'] = truncnorm(0, np.inf, amp, amp*0.1)
        self.parameter_guesses['shift'] = norm(0, 0.01)

    def add_absorption_line(self, amplitude=None, width=None, shift=0, name=None):
        if name is None:
            name = '{}_absorp'.format(self.name)
        return self.parent_region.add_feature(name, PairedAbsorptionLine, self, centre=self.parameters['centre'].get_value(),
                                       amplitude=amplitude, width=width, shift=shift)


class AbsorptionLine(SpectralFeature):
    """AbsorptionLine is defined with a positive amplitude"""
    default_conditions = {'amplitude': partial(T.lt, 0), 'width': partial(T.lt, 0)}
    derived_quantities = {'flux': [lambda *a: -line_flux(*a), ('amplitude', 'width')]}

    def __init__(self, name, centre=None, amplitude=None, width=None, shift=0, parent_region=None):
        super(AbsorptionLine, self).__init__(name, gaussian_absorption_line, [centre, amplitude, width, shift], parent_region)

    def guess(self, wavelength, flux, error, smoothing_window):
        median_smoothed = smooth_spectrum(flux, wavelength, smoothing_window)
        filter = self.parent_region.filter(wavelength)
        w, smoothf, e = wavelength[filter], median_smoothed[filter], error[filter]
        cent, std, amp = characterise_peak(w, smoothf-flux[filter], np.std(smoothf)*2, self.parameters['centre'].get_value())
        self.parameter_guesses['width'] = truncnorm(0, np.inf, std, std*0.1)
        self.parameter_guesses['amplitude'] = truncnorm(0, np.inf, amp, amp*0.1)
        self.parameter_guesses['shift'] = norm(0, 0.01)


class PairedAbsorptionLine(AbsorptionLine):
    def __init__(self, name, emission_line, centre=None, amplitude=None, width=None, shift=0, parent_region=None):
        super(PairedAbsorptionLine, self).__init__(name, centre, amplitude, width, shift, parent_region)
        self.paired_line = emission_line

    def guess(self, wavelength, flux, error, smoothing_window):
        class stat(object):
            def __init__(self, other_stat):
                self.other_stat = other_stat
                self.mul = 1

            def rvs(self, *args, **kwargs):
                return self.other_stat.rvs(*args, **kwargs) * self.mul

            def __mul__(self, other):
                self.mul *= other
                return self

        self.parameter_guesses['width'] = stat(self.paired_line.parameter_guesses['width']) * 2.5
        self.parameter_guesses['amplitude'] = stat(self.paired_line.parameter_guesses['amplitude']) * 0.01
        self.parameter_guesses['shift'] = self.paired_line.parameter_guesses['shift']


class Continuum(SpectralFeature):
    def __init__(self, name, level=None, parent_region=None):
        def cont(x, level):
            return (x/x) * level
        super(Continuum, self).__init__(name, cont , [level], parent_region)

    def guess_level(self, wavelength, flux, error, smoothing_window):
        median_smoothed = smooth_spectrum(flux, wavelength, smoothing_window)
        cont_med, cont_std = np.median(median_smoothed[self.parent_region.filter(wavelength)]), np.std(
            median_smoothed[self.parent_region.filter(wavelength)])
        return norm(cont_med, cont_std * 0.5)


class SpectrumRegion(object):
    def __init__(self, model, min, max, name):
        self.model = model
        self.min = min
        self.max = max
        self.name = name

        self.data_wavelength = None
        self.data_flux = None
        self.data_error = None
        self.features = []

    def add_emission_line(self, name, centre, amplitude=None, width=None, shift=0):
        return self.add_feature(name, EmissionLine, centre=centre, amplitude=amplitude, width=width, shift=shift)

    def add_absorption_line(self, name, centre, amplitude=None, width=None, shift=0):
        return self.add_feature(name, AbsorptionLine, centre=centre, amplitude=amplitude, width=width, shift=shift)

    def add_continuum(self, level=None):
        return self.add_feature('{}_continuum'.format(self.name), Continuum, level=level)

    def add_feature(self, name, feature_class, *feature_args, **parameters):
        params = {}
        for pname, pvalue in parameters.iteritems():
            parameter_name = '{}_{}'.format(name, pname)
            if pvalue is None:
                p = T.dscalar(parameter_name)
                if not self.model._theano_variable_recorded(p):
                    self.model._record_var(p, self.name, name, pname)
            elif isinstance(pvalue, (T.TensorConstant, T.TensorVariable)):
                p = pvalue
                if not self.model._theano_variable_recorded(p):
                    self.model._record_var(p, self.name, name, pname)
            else:
                p = shared(pvalue, parameter_name)
                self.model._theano_shared.append(p)
            params[pname] = p
        feature = feature_class(name, *feature_args, parent_region=self, **params)
        self.features.append(feature)
        self.model.theano_feature_model = None  # adding new feature erases old model
        return feature

    def feature_model(self, w):
        return T.sum([feature.function(w) for feature in self.features], axis=0)

    def filter(self, w):
        return (w >= self.min) & (w <= self.max)

    def loglikelihood(self, w, f, e):
        """
        returns the theano function taking wavelength, flux, and error arrays followed by float parameters in the order of self.parameters
        """
        filter = self.filter(w).nonzero()
        return lnlike(f[filter], e[filter], self.feature_model(w[filter]))

    def build_model(self):
        w = T.dvector('wavelengths')
        pars = [p for feature in self.features for p in feature.parameters.values() if p in self.model._theano_vars]
        return tfunction([w]+pars, self.feature_model(w)), [p.name for p in pars]


class SpectrumModel(object):
    def __init__(self):
        self.regions = []
        self._theano_vars = []
        self._theano_var_directory = []
        self._theano_shared = []

        self._theano_wvl = T.dvector('data_wavelength')
        self._theano_flx = T.dvector('data_flux')
        self._theano_err = T.dvector('data_error')

        self.var_conditions = {}
        self.reset_conditions()
        self.wavelength_shift = 0

    def allow_shift(self, limit=5):
        self.wavelength_shift = T.dscalar('wavelength_shift')
        self._record_var(self.wavelength_shift, 'global', 'shift', 'delta')
        self.add_conditions(T.abs_(self.wavelength_shift) < limit)

    @property
    def arguments(self):
        return self._theano_data+self._theano_vars

    @property
    def argument_names(self):
        return [i.name for i in self.arguments]

    @property
    def _theano_data(self):
        return [self._theano_wvl, self._theano_flx, self._theano_err]

    @property
    def features(self):
        return [f for region in self.regions for f in region.features]

    def _theano_variable_recorded(self, v):
        return any(i in self._theano_vars for i in get_parameter_dependencies(v)) or v in self._theano_vars

    def _record_var(self, v, region_name, feature_name, parameter_name):
        self._theano_var_directory.append((region_name, feature_name, parameter_name))
        self._theano_vars.append(v)

    @property
    def parameter_names(self):
        return [i.name for i in self._theano_vars]

    def add_region(self, min, max, name=None):
        if name is None:
            name = 'region_{}'.format(len(self.regions))
        self.regions.append(SpectrumRegion(self, min, max, name))
        return self.regions[-1]

    def auto_conditions(self):
        for f in self.features:
            for n, p in f.parameters.iteritems():
                try:
                    self.add_conditions(f.default_conditions[n](p))
                except KeyError:
                    pass

    def add_conditions(self, *conditions):
        """use to make more complex conditions not effecting the guessing"""
        assert all(self._theano_variable_recorded(v) for v in conditions), "one or more boundaries are not based on valid parameters"
        self.var_conditions += conditions

    def reset_conditions(self):
        self._theano_conditions = shared(1, 'conditions')
        self.var_conditions = []

    def build_conditions(self):
        for cond in self.var_conditions:
            self._theano_conditions &= cond

    def loglikelihood(self):
        w, f, e = self._theano_data
        if self.wavelength_shift != 0:
            w += self.wavelength_shift
        return ifelse(self._theano_conditions,
                      T.sum([r.loglikelihood(w, f, e) for r in self.regions]),
                      ninf)

    def build_loglikelihood(self):
        return tfunction(self.arguments, self.loglikelihood())

    def build_sampler(self, data_wavelength, data_flux, data_error, nwalkers, **emcee_parameters):
        theano_func = self.build_loglikelihood()
        func = lambda theta: theano_func(data_wavelength, data_flux, data_error, *theta)
        self.sampler = emcee.EnsembleSampler(nwalkers, len(self._theano_vars), func, **emcee_parameters)
        for f in self.features:
            for pname, p in f.parameters.iteritems():
                try:
                    f._chain_index[pname] = self._theano_vars.index(p)
                except ValueError:
                    pass


    def get_guesses(self, nwalkers):
        l = []
        for guess in self.ordered_parameter_guesses:
            l.append(guess)
        return l

    @property
    def ordered_parameter_guesses(self):
        l = []
        region_names = [i.name for i in self.regions]
        for r, f, p in self._theano_var_directory:
            if r != 'global':
                region = self.regions[region_names.index(r)]
                feature_names = [i.name for i in region.features]
                feature = region.features[feature_names.index(f)]
                try:
                    guess = feature.parameter_guesses[p]
                    l.append(guess)
                except KeyError:
                    raise ValueError("{}_{} ({}) has not been guessed yet".format(f, p, r))
        if self.wavelength_shift != 0:
            l.append(self.wavelength_shift_guess)
        return l

    @property
    def all_guessed(self):
        return len(self.ordered_parameter_guesses) == len(self._theano_vars)

    def build(self, data_wavelength, data_flux, data_error, nwalkers):
        self.build_conditions()
        self.build_sampler(data_wavelength, data_flux, data_error, nwalkers)
        self.data_wavelength = data_wavelength
        self.data_flux = data_flux
        self.data_error = data_error

    def fit(self, data_wavelength, data_flux, data_error, nsteps, nwalkers, **emcee_parameters):
        if not self.all_guessed:
            raise ValueError("Some parameters have not been guessed at")
        self.build(data_wavelength, data_flux, data_error, nwalkers)
        p0 = np.asarray([g.rvs(nwalkers) for g in self.ordered_parameter_guesses]).T
        for i, pname in enumerate(self.parameter_names):
            print 'guessing {} between {:.2f} and {:.2f}'.format(pname, p0[:, i].min(), p0[:, i].max())
        for i in tqdm(self.sampler.sample(p0, iterations=nsteps, **emcee_parameters), total=nsteps):
            pass

    def auto_guess(self, wavelength, flux, error, smoothing_window):
        if self.wavelength_shift != 0:
            self.wavelength_shift_guess = norm(0, 0.01)
        for r in self.regions:
            for f in r.features:
                f.guess(wavelength, flux, error, smoothing_window)

    @property
    def region_models(self):
        if self.theano_feature_model is None:
            w = T.dvector('wavelength')
            outputs = [r.feature_model(w + self.wavelength_shift) for r in self.regions]
            self.theano_feature_model = tfunction([w]+self._theano_vars, outputs)
        def inner(x, *args):
            ys = self.theano_feature_model(x, *args)
            filts = [r.filter(x) for r in self.regions]
            return [(x[filts[i]], ys[i][filt]) for i, filt in enumerate(filts)]
        return inner

    def plot_model(self, x, *args, **kwargs):
        rs = self.region_models(x, *args)
        y = kwargs.get('y', None)
        yerr = kwargs.get('yerr', None)
        if kwargs.get('separate_region_axes', False):
            fig, axes = plt.subplots(1, len(rs))
            for ax, r in zip(axes, rs):
                ax.plot(r[0], r[1], 'r-', label='model')
                if y is not None:
                    _y = y[(x <= r[0].max()) & (x >= r[0].min())]
                    ax.plot(r[0], _y, 'k--', label='data')
                if yerr is not None:
                    _e = yerr[(x <= r[0].max()) & (x >= r[0].min())]
                    ax.fill_between(r[0], _y-_e, _y+_e, alpha=0.2, color='k')
        else:
            fig, ax = plt.subplots()
            for r in rs:
                ax.plot(r[0], r[1], 'r-', label='model')
                if y is not None:
                    _y = y[(x <= r[0].max()) & (x >= r[0].min())]
                    ax.plot(r[0], y, 'k--', label='data')
                if yerr is not None:
                    _e = yerr[(x <= r[0].max()) & (x >= r[0].min())]
                    ax.fill_between(r[0], _y-_e, _y+_e, alpha=0.2, color='k')
        axes[0].legend()
        return fig

    def plot_fit(self, percentiles=(16, 50, 84), burnin=None, thin=1, separate_region_axes=True, text=False):
        if burnin is None:
            burnin = int(self.sampler.chain.shape[1] * 0.5)
            warn("no burnin given, using burnin of 50%")
        fitted = np.percentile(self.sampler.chain[:, burnin::thin].reshape(-1, self.sampler.chain.shape[-1]), percentiles, axis=0)

        lines = [(f.parameters['centre'].get_value(), f.name) for f in self.features if 'centre' in f.parameters]
        x = self.data_wavelength
        low, med, high = [self.region_models(x, *f) for f in fitted]
        if separate_region_axes:
            fig, (axes, res_axes) = plt.subplots(2, len(med), sharex='col')
            for ax, res_ax, r_low, r_med, r_high in zip(axes, res_axes, low, med, high):
                ax.plot(r_med[0], r_med[1], 'r-', label='model')
                ax.fill_between(r_low[0], r_low[1], r_high[1], alpha=0.2, color='r')
                if self.data_flux is not None:
                    _y = self.data_flux[(x <= r_med[0].max()) & (x >= r_med[0].min())]
                    ax.plot(r_med[0], _y, 'k-', label='data')
                    res_ax.plot(r_med[0], _y - r_med[1], 'r-')
                    res_ax.fill_between(r_low[0], _y - r_low[1], _y - r_high[1], alpha=0.2, color='r')
                if self.data_error is not None:
                    _e = self.data_error[(x <= r_med[0].max()) & (x >= r_med[0].min())]
                    ax.fill_between(r_med[0], _y - _e, _y + _e, alpha=0.2, color='k')
                    res_ax.fill_between(r_med[0], -_e, _e, alpha=0.2, color='k')
                line_wave, line_lab = zip(*[l for l in lines if r_med[0].min() <= l[0] <= r_med[0].max()])
                lineid_plot.plot_line_ids(r_low[0], r_med[1], line_wave, line_lab, ax=ax)
                for l in line_wave:
                    res_ax.axvline(l, ls='--', color='k')


        else:
            fig, (ax, res_ax) = plt.subplots(2, sharex='col')
            for r_low, r_med, r_high in zip(low, med, high):
                ax.plot(r_med[0], r_med[1], 'r-', label='model')
                ax.fill_between(r_low[0], r_low[1], r_high[1], alpha=0.2, color='r')
                if self.data_flux is not None:
                    _y = self.data_flux[(x <= r_med[0].max()) & (x >= r_med[0].min())]
                    ax.plot(r_med[0], _y, 'k-', label='data')
                    res_ax.plot(r_med[0], _y - r_med[1], 'r-')
                    res_ax.fill_between(r_low[0], _y - r_low[1], _y - r_high[1], alpha=0.2, color='r')
                if self.data_error is not None:
                    _e = self.data_error[(x <= r_med[0].max()) & (x >= r_med[0].min())]
                    ax.fill_between(r_med[0], _y - _e, _y + _e, alpha=0.2, color='k')
                    res_ax.fill_between(r_med[0], -_e, _e, alpha=0.2, color='k')
                line_wave, line_lab = zip(*[l for l in lines if r_med[0].min() <= l[0] <= r_med[0].max()])
                lineid_plot.plot_line_ids(r_low[0], r_med[1], line_wave, line_lab, ax=ax)
                for l in line_wave:
                    res_ax.axvline(l, ls='--', color='k')


        if text:
            s = '\n'.join(['{}$={:.2f}\pm{:.2f}$'.format(n, f[0], (f[2] - f[1])/2.) for n, f in zip(self.parameter_names, fitted.T)])
            ax.text(0.99, 0.99, s, transform=ax.transAxes, ha='right', va='top')

        fig.subplots_adjust(hspace=0)
        return fig

    def plot_chains(self, names=None):
        if names is None:
            names = self.parameter_names
        inds = [self.parameter_names.index(n) for n in names]
        fig, axes = plt.subplots(len(inds))
        for n, i, ax in zip(names, inds, axes):
            ax.plot(self.sampler.chain[:,:,i].T)
            ax.set_ylabel(n)
        return fig

    def plot_corner(self, burnin=None):
        if burnin is None:
            burnin = self.sampler.chain.shape[1] * 0.5
            warn("no burnin given, using burnin of 50%")
        s = self.sampler.chain[:, burnin:, :].reshape(-1, self.sampler.chain.shape[-1])
        return corner(s, labels=self.parameter_names)

    def fitted_parameters(self, burnin=None, thin=1):
        if burnin is None:
            burnin = self.sampler.chain.shape[1] * 0.5
            warn("no burnin given, using burnin of 50%")
        fitted = np.percentile(self.sampler.chain[:, burnin::thin].reshape(-1, self.sampler.chain.shape[-1]),
                               (16, 50, 84), axis=0)
        return pd.DataFrame(fitted.T, columns=['lower', 'median', 'upper'], index=self.parameter_names)


if __name__ == '__main__':
    from astroML.datasets import fetch_sdss_spectrum
    spec = fetch_sdss_spectrum(1975, 53734, 1).restframe()
    x, y, e = spec.wavelength(), spec.spectrum, spec.error

    norm_const = np.median(spec.spectrum[(x >= 5500) & (x <= 5600)])
    y /= norm_const
    e /= norm_const

    model = SpectrumModel()
    ha_group = model.add_region(6450, 6775, 'halpha')
    hb_group = model.add_region(4800, 5050, 'hbeta')

    NIIa = ha_group.add_emission_line('NIIa', 6549.86)
    NIIb = ha_group.add_emission_line('NIIb', 6585.27, amplitude=NIIa.amplitude/0.34, width=NIIa.width)
    SIIa = ha_group.add_emission_line('SIIa', 6718.29, width=NIIa.width)
    SIIb = ha_group.add_emission_line('SIIb', 6732.68, width=NIIa.width)

    OIII5007 = hb_group.add_emission_line('OIII5007', 5008.239, width=NIIa.width)
    OIII4958 = hb_group.add_emission_line('OIII4958', 4960.295, amplitude=OIII5007.amplitude*0.35, width=NIIa.width)

    Hb = hb_group.add_emission_line('Hb', 4862.7, width=NIIa.width)
    Ha = ha_group.add_emission_line('Ha', 6564.614, width=NIIa.width, amplitude=Hb.amplitude/0.35)

    Ha.add_absorption_line()
    Hb.add_absorption_line()

    ha_group.add_continuum()
    hb_group.add_continuum()
    model.allow_shift(limit=1)


    model.auto_conditions()
    model.add_conditions(NIIa.width < 100)
    model.auto_guess(x, y, e, 200)
    model.fit(x, y, e, 100, 200)
    model.plot_fit()
    fitted = model.fitted_parameters().T.iloc[1]
    print fitted

    nii = line_flux(fitted.NIIa_amplitude / 0.34, fitted.NIIa_width)
    ha = line_flux(fitted.Hb_amplitude / 0.35, fitted.NIIa_width)
    hb = line_flux(fitted.Hb_amplitude, fitted.NIIa_width)
    oiii = line_flux(fitted.OIII5007_amplitude, fitted.NIIa_width)
    print [nii/ha, oiii/hb]
    print [167.8/432.4, 27.6/70.55]
    # model.plot_chains()
    # model.plot_corner()



    plt.show()