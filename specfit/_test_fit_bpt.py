from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import emcee
from tqdm import tqdm
import lineid_plot
from scipy.signal import medfilt
from corner import corner

NII_amplitude_ratio_6548_6584 = 0.34  # oh 2011
OIII_amplitude_ratio_4958_5007 = 0.35
Hb_Ha_amplitude_ratio = 0.35


NIIa = 6549.86
NIIb = 6585.27
Halpha = 6564.614
SIIa = 6718.29
SIIb = 6732.68
Hbeta = 4862.7
OIII5007 = 5008.239
OIII4958 = 4960.295

halpha_complex_region = [6450, 6775]
hbeta_complex_region = [4800, 5050]

MAX_SHIFT = 1

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

def fwhm(std):
    return 2.3548 * std

def equivalent_width():
    """assuming a uniform continuum"""
    return line_flux / continuum_flux

def gaussian_emission_line(amplitude, centre, std, x):
    return amplitude * np.exp(-1. * ((x - centre)**2.) / (2*std*std))

def bpt_lines_model(theta, ha_complex_wavelengths, hb_complex_wavelengths):
    if len(theta) == 14:
        ha_amp, niib_amp, siia_amp, siib_amp, hb_amp, oiii5007_amp = theta[:6]  
        line_width = theta[6]
        halpha_cont, hbeta_cont, shift = theta[7:-4]
    elif len(theta) == 13:
        ha_amp, niib_amp, siia_amp, siib_amp, oiii5007_amp = theta[:5]  
        line_width = theta[5]
        halpha_cont, hbeta_cont, shift = theta[6:-4]
        hb_amp = Hb_Ha_amplitude_ratio * ha_amp
    else:
        raise ValueError("length of theta is not 11 or 12")
    hb_abs_amp, hb_abs_width = theta[-4:-2]
    ha_abs_amp, ha_abs_width = theta[-2:]

    niia_flux = gaussian_emission_line(niib_amp * NII_amplitude_ratio_6548_6584, NIIa, line_width, ha_complex_wavelengths - shift)
    niib_flux = gaussian_emission_line(niib_amp, NIIb, line_width, ha_complex_wavelengths - shift)
    ha_flux = gaussian_emission_line(ha_amp, Halpha, line_width, ha_complex_wavelengths - shift)
    siia_flux = gaussian_emission_line(siia_amp, SIIa, line_width, ha_complex_wavelengths - shift)
    siib_flux = gaussian_emission_line(siib_amp, SIIb, line_width, ha_complex_wavelengths - shift)

    oiii5007_flux = gaussian_emission_line(oiii5007_amp, OIII5007, line_width, hb_complex_wavelengths - shift)
    oiii4958_flux = gaussian_emission_line(oiii5007_amp * OIII_amplitude_ratio_4958_5007, OIII4958, line_width, hb_complex_wavelengths - shift)
    hb_flux = gaussian_emission_line(hb_amp, Hbeta, line_width, hb_complex_wavelengths - shift)
    
    hb_abs = gaussian_emission_line(hb_abs_amp, Hbeta, hb_abs_width, hb_complex_wavelengths-shift)
    ha_abs = gaussian_emission_line(ha_abs_amp, Halpha, ha_abs_width, ha_complex_wavelengths-shift)

    ha_complex_model = niia_flux + niib_flux + ha_flux + siia_flux + siib_flux + halpha_cont + ha_abs
    hb_complex_model = oiii5007_flux + oiii4958_flux + hb_flux + hbeta_cont + hb_abs
    return ha_complex_model, hb_complex_model

def lnlike_lines(theta, ha_complex_flux, ha_complex_wavelengths, ha_complex_error, hb_complex_flux, hb_complex_wavelengths, hb_complex_error):
    ha_complex_model, hb_complex_model = bpt_lines_model(theta, ha_complex_wavelengths, hb_complex_wavelengths)
    ha_residuals = ((ha_complex_model - ha_complex_flux) / ha_complex_error)**2.
    hb_residuals = ((hb_complex_model - hb_complex_flux) / hb_complex_error)**2.
    return -0.5 * (np.sum(ha_residuals) + np.sum(hb_residuals))
  

def lnprob_lines(theta, ha_complex_flux, ha_complex_wavelengths, ha_complex_error,
                 hb_complex_flux, hb_complex_wavelengths, hb_complex_error):
    shift = theta[-5]
    if abs(shift) > MAX_SHIFT:
        return -np.inf
    if not np.all(theta[:len(theta) - 7] >= 0):
        # all amps > 0
        return -np.inf
    if theta[-6] > 100:
        return -np.inf
    if theta[-1] > 100:
        return -np.inf
    if theta[-2] > 0:
        return -np.inf
    if theta[-3] > 100:
        return -np.inf
    if theta[-4] > 0:
        return -np.inf
    return lnlike_lines(theta, ha_complex_flux, ha_complex_wavelengths, ha_complex_error,
                 hb_complex_flux, hb_complex_wavelengths, hb_complex_error)


def characterise_peak(wavelengths, continuum_subtracted_flux, continuum_error, continuum_subtracted_err, expected_peak_wavelength):
    """returns basic fit [lambda, std, amplitude] for a expected peak. Returns a stupidly small peak if not found"""
    regions = contiguous_regions(continuum_subtracted_flux > continuum_error)    
    regions = regions[(regions[:, 1] - regions[:, 0]) > 1][:-1]  # extended regions
    region_wvls = wavelengths[regions]
    peak_region = region_wvls[(region_wvls[:, 1] >= expected_peak_wavelength) & (region_wvls[:, 0] <= expected_peak_wavelength)]
    if len(peak_region) == 0:
        print 'peak finding failed for', expected_peak_wavelength, 'assuming no peak'
        return [expected_peak_wavelength, 0.001, 0.]
    peak_region = peak_region[0]
    filt = (wavelengths >= peak_region[0]) & (wavelengths <= peak_region[1])
    
    from scipy.optimize import curve_fit
    guess = [expected_peak_wavelength, 0.5*(peak_region[1] - peak_region[0]), 1.]

    popt, pcov = curve_fit(lambda x, *p: gaussian_emission_line(*p, x=x), wavelengths[filt], continuum_subtracted_flux[filt], guess)

    plt.plot(wavelengths, continuum_subtracted_flux)
    print popt
    plt.plot(wavelengths[filt], gaussian_emission_line(*popt, x=wavelengths[filt]), 'r--')
    plt.show()
    return popt


def characterise_peaks(wavelengths, continuum_subtracted_flux, continuum_error, continuum_subtracted_err, expected_peak_wavelengths):
    l = []
    for w in expected_peak_wavelengths:
        popt = characterise_peak(wavelengths, continuum_subtracted_flux, continuum_error, continuum_subtracted_err, w)
        l.append(popt)
    return np.asarray(l)


def smooth_spectrum(flux, wavelength, wvl_window=200):
    n = (wavelength[-1] - wavelength[0]) / len(wavelength)  # ang / block
    window = (int(wvl_window / n / 2) * 2) + 1  # always have an odd window length
    return medfilt(flux, window)


def fit_bpt_lines(wavelength, flux, error, nwalkers=500, nsteps=1000, link_hlines=True):
    """
    Assumes the emission lines come from the same region (same widths)

    fit with free parameters:
        NIIb, OIII5007, Ha amplitudes
        line width
        one uniform continuum for each complex

    theta is defined as:
        Ha_amplitude
        NIIb_amplitude
        SIIa_amplitude
        SIIb_amplitude 
        Hb_amplitude
        OIII5007_amplitude
        line_width
        halpha_complex_continuum
        hbeta_complex_continuum
        wavelength_shift
        hbeta abs amp
        hbeta abs width
        halpha abs amp
        halpha abs width
    """
    halpha_complex_filter = (wavelength >= halpha_complex_region[0]) & (wavelength <= halpha_complex_region[1])
    hbeta_complex_filter = (wavelength >= hbeta_complex_region[0]) & (wavelength <= hbeta_complex_region[1])

    median_smoothed = smooth_spectrum(flux, wavelength)
    halpha_med, halpha_std = np.median(median_smoothed[halpha_complex_filter]), np.std(median_smoothed[halpha_complex_filter])
    hbeta_med, hbeta_std = np.median(median_smoothed[hbeta_complex_filter]), np.std(median_smoothed[hbeta_complex_filter])

    halpha_cont_guess = np.random.normal(halpha_med, halpha_std * 0.5, nwalkers)
    hbeta_cont_guess = np.random.normal(hbeta_med, hbeta_std * 0.5, nwalkers)

    ha_peaks = characterise_peaks(wavelength[halpha_complex_filter], flux[halpha_complex_filter] - halpha_med, halpha_std, error[halpha_complex_filter] - halpha_med, 
                                [NIIa, NIIb, Halpha, SIIa, SIIb])
    hb_peaks = characterise_peaks(wavelength[hbeta_complex_filter], flux[hbeta_complex_filter] - halpha_med, hbeta_std, error[hbeta_complex_filter] - hbeta_med, 
                                [Hbeta, OIII5007, OIII4958])
    
    width_mean = np.mean([np.mean(ha_peaks[:, 1]), np.mean(hb_peaks[:-1, 1])])  # not OIII4958
    width_guess = stats.truncnorm(0, 100, width_mean, width_mean * 0.1).rvs(nwalkers)

    niib_amp_guess = stats.truncnorm(0, np.inf, ha_peaks[1, 1], ha_peaks[1, 1] * 0.1).rvs(nwalkers)
    halpha_amp_guess = stats.truncnorm(0, np.inf, ha_peaks[2, 1], ha_peaks[2, 1] * 0.1).rvs(nwalkers)
    siia_amp_guess = stats.truncnorm(0, np.inf, ha_peaks[3, 1], ha_peaks[3, 1] * 0.1).rvs(nwalkers)
    siib_amp_guess = stats.truncnorm(0, np.inf, ha_peaks[4, 1], ha_peaks[4, 1] * 0.1).rvs(nwalkers)

    hbeta_amp_guess = stats.truncnorm(0, np.inf, hb_peaks[0, 1], hb_peaks[0, 1] * 0.1).rvs(nwalkers)
    oiii5007_amp_guess = stats.truncnorm(0, np.inf, hb_peaks[1, 1], hb_peaks[1, 1] * 0.1).rvs(nwalkers)

    shift_guess = np.random.normal(0, 0.01, nwalkers)

    hbeta_abs_amp = hbeta_amp_guess * -0.1  # smaller absoprtion line amp
    hbeta_abs_width = width_guess * 1.5  # wider
    hbeta_abs_width[hbeta_abs_width > 100] = 100

    halpha_abs_amp = halpha_amp_guess * -0.1
    halpha_abs_width = width_guess * 1.5  # wider
    halpha_abs_width[halpha_abs_width > 100] = 100

    if not link_hlines:
        labs = [r'$H\alpha$ amp', '$NIIb$ amp', '$SIIa$ amp', '$SIIb$ amp', r'$H_\beta$ amp', '$OIII[5007]$ amp', 
        r'$\sigma$', r'$cont_{H\alpha}$', r'$cont_{H\beta}$', r'$\Delta \lambda$',
        r'$H\beta$ abs amp', r'$H\beta$ abs width', r'$H\alpha$ abs amp', r'$H\alpha$ abs width']
        guess = np.stack((halpha_amp_guess, niib_amp_guess, siia_amp_guess, siib_amp_guess, hbeta_amp_guess, oiii5007_amp_guess, 
                    width_guess, halpha_cont_guess, hbeta_cont_guess, shift_guess, hbeta_abs_amp, hbeta_abs_width, halpha_abs_amp, halpha_abs_width))
    else:
        labs = [r'$H\alpha$ amp', '$NIIb$ amp', '$SIIa$ amp', '$SIIb$ amp', '$OIII[5007]$ amp',
        r'$\sigma$', r'$cont_{H\alpha}$', r'$cont_{H\beta}$', r'$\Delta \lambda$',
        r'$H\beta$ abs amp', r'$H\beta$ abs width', r'$H\alpha$ abs amp', r'$H\alpha$ abs width']
        guess = np.stack((halpha_amp_guess, niib_amp_guess, siia_amp_guess, siib_amp_guess, oiii5007_amp_guess, 
                    width_guess, halpha_cont_guess, hbeta_cont_guess, shift_guess, hbeta_abs_amp, hbeta_abs_width, halpha_abs_amp, halpha_abs_width))
    # plot_fit(wavelength, flux, error, guess.T, 0)
    # plt.show()

    sampler = emcee.EnsembleSampler(
        nwalkers, len(guess), lnprob_lines, 
        args=(flux[halpha_complex_filter], wavelength[halpha_complex_filter], error[halpha_complex_filter],
              flux[hbeta_complex_filter], wavelength[hbeta_complex_filter], error[hbeta_complex_filter]))

    for i in tqdm(sampler.sample(guess.T, iterations=nsteps), total=nsteps):
        pass
    return sampler, labs


def plot_fit(wavelength, flux, error, chain, nburnin):
    halpha_complex_filter = (wavelength >= halpha_complex_region[0]) & (wavelength <= halpha_complex_region[1])
    hbeta_complex_filter = (wavelength >= hbeta_complex_region[0]) & (wavelength <= hbeta_complex_region[1])

    parameters = np.percentile(chain[:, nburnin:].reshape(-1, chain.shape[-1]), [16, 50, 84], axis=0)
    
    fig, axes = plt.subplots(2, 2, sharex='col')
    plt.subplots_adjust(wspace=0, hspace=0)

    models = bpt_lines_model(parameters[1], wavelength[halpha_complex_filter], wavelength[hbeta_complex_filter])

    line_wvls = [[NIIa, NIIb, Halpha, SIIa, SIIb], [Hbeta, OIII4958, OIII5007]]
    line_labels = [['$NII[6548]$', '$NII[6584]$', r'$H\alpha$', '$SIIa$', '$SIIb$'], [r'$H\beta$', '$OIII[4958]$', '$OIII[5007]$']]

    median_smoothed = smooth_spectrum(flux, wavelength)

    for ax, res_ax, filt, model, line_wvl, line_lab, cont in zip(axes[0], axes[1], 
                                                          [halpha_complex_filter, hbeta_complex_filter],
                                                          models, line_wvls, line_labels, 
                                                          [parameters[:,-3], parameters[:,-2]]):
        wvl = wavelength[filt]
        flx = flux[filt]
        flx_upper = flx + error[filt]
        flx_lower = flx - error[filt]
        ax.plot(wvl, flx, 'k-')
        ax.fill_between(wvl, flx_lower, flx_upper, color='k', alpha=0.3)
        ax.plot(wvl, model, 'r-')

        res_ax.plot(wvl, flx - model, 'k-')
        res_ax.fill_between(wvl, flx_lower-model, flx_upper-model, color='k', alpha=0.3)

        lineid_plot.plot_line_ids(wvl, flx, line_wvl, line_lab, ax=ax)
        for w in line_wvl:
            res_ax.axvline(x=w, linestyle='--', color='r')

        cont_med, cont_std = np.median(median_smoothed[filt]), np.std(median_smoothed[filt])
        ax.axhline(cont_med, color='g', ls=':')
        res_ax.axhline(0, color='g', ls=':')
        ax.axhspan(cont_med-cont_std, cont_med+cont_std, alpha=0.2, color='g')
        res_ax.axhspan(-cont_std, cont_std, color='g', alpha=0.2)
    return fig

def plot_chain(chain, labels=None, nstepthin=5, nwalkerthin=1):
    fig, ax = plt.subplots(chain.shape[-1], sharex=True)
    for i, a in enumerate(ax):
        a.plot(chain[::nwalkerthin, ::nstepthin, i].T)
        if labels is not None:
            a.set_ylabel(labels[i])

def plot_params(chain, nburnin, labels):
    corner(chain[:, nburnin:].reshape(-1, chain.shape[-1]), labels=labels)


if __name__ == '__main__':
    # from astroML.datasets import fetch_sdss_spectrum
    # spec = fetch_sdss_spectrum(1975, 53734, 1).restframe()
    # norm = np.median(spec.spectrum[(spec.wavelength() >= 5500) & (spec.wavelength() <= 5600)])

    # w, f, e = spec.wavelength(), spec.spectrum, spec.error


    # sampler, labels = fit_bpt_lines(w, f, e, nwalkers=50, nsteps=2000, link_hlines=True)

    # burnin = 1000
    # plot_chain(sampler.chain, labels, 1, 1)
    # # plot_params(sampler.chain, burnin, labels)
    # plot_fit(w, f, e, sampler.chain, burnin)
    # plt.show()

    class A(object):
        def __init__(self, a, b=None):
            self.a = a
            self.b = b

        

    a_cls = A(1)
    b_cls = A(2, a_cls.a)

    a_cls.a = 10
    print a_cls.a, a_cls.b
    print b_cls.a, b_cls.b