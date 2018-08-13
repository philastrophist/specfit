import spectrum_model
import numpy as np

def bpt_model():
    """restframe bpt model"""
    model = spectrum_model.SpectrumModel()
    ha_group = model.add_region(6525, 6775, 'halpha')
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
    model.allow_shift(limit=10)

    model.auto_conditions()
    model.add_conditions(NIIa.width < 100)
    return model

def fit_bpt(x, y, e, nwalkers, nsteps, window=200):
    """
    fit bpt lines to a restframe spectrum
    returns the line fluxes, line ratios, and the model (for diagnostics and plotting)
    """
    model = bpt_model()
    model.auto_guess(x, y, e, window)
    model.fit(x, y, e, nsteps, nwalkers)
    fitted = model.fitted_parameters().T

    nii = spectrum_model.line_flux(fitted.NIIa_amplitude / 0.34, fitted.NIIa_width)
    ha = spectrum_model.line_flux(fitted.Hb_amplitude / 0.35, fitted.NIIa_width)
    hb = spectrum_model.line_flux(fitted.Hb_amplitude, fitted.NIIa_width)
    oiii = spectrum_model.line_flux(fitted.OIII5007_amplitude, fitted.NIIa_width)

    nii = nii.iloc[1], nii.iloc[2] - nii.iloc[0]
    ha = ha.iloc[1], ha.iloc[2] - ha.iloc[0]
    hb = hb.iloc[1], hb.iloc[2] - hb.iloc[0]
    oiii = oiii.iloc[1], oiii.iloc[2] - oiii.iloc[0]

    ratios = [np.log10(nii[0]/ha[0]), np.log10(oiii[0]/hb[0])]
    ratio_errs = [0.434*np.sqrt((nii[1]/nii[0])**2 + (ha[1]/ha[0])**2), 0.434*np.sqrt((oiii[1]/oiii[0])**2 + (hb[1]/hb[0])**2)]
    return ratios, ratio_errs, model

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # from astroML.datasets import fetch_sdss_spectrum
    # spec = fetch_sdss_spectrum(1975, 53734, 1).restframe()
    # x, y, e = spec.wavelength(), spec.spectrum, spec.error

    file = np.load('../../TheFIRC4/stacked_agn_spectrum.npz')
    x, y, e = file['wvl'], file['flux'], file['err']

    plt.plot(x, y)
    plt.show()


    # model = bpt_model()
    # model.auto_guess(x, y, e, 200)
    # model.fit(x, y, e, 1000, 100)
    # model.plot_fit()
    # spectrum_model.plt.show()