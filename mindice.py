# mindice library, written by luisgdh
# https://sites.google.com/site/luisgdh
#
# V 1.1
# 
# mindice(wl, flx, err, 'CO2.30', definitions = definitions)
#
# Changes:
# 2026.03.23 Added an example code with a generic optical spectrum
# 
from matplotlib import pyplot as plt
import numpy as np

def mindice(wl, flx, err = None, ind = None, coeff = 1, plot = False,
            definitions = None):
    """
    Measure spectral indices (Equivalent Width or Magnitudes) for stellar/galactic spectra.

    This function performs a local continuum normalization using a polynomial fit 
    across multiple user-defined bandpasses and integrates the flux within 
    specified feature windows. It supports both Angstrom (EW) and Magnitude 
    (Lick-style) units, handling multiple feature bands for a single index.

    Parameters:
    -----------
    wl : array-like
        Wavelength array (must be strictly increasing).
    flx : array-like
        Flux array, same length as wl.
    err : array-like, optional
        1-sigma error array for flux. If provided, the function returns 
        propagated uncertainties (EWe).
    ind : str
        The name of the index to measure (must exist in definitions).
    coeff : int, default 1
        The degree of the polynomial fit for the continuum (1 = linear).
    plot : bool, default False
        If True, generates a two-panel diagnostic plot showing the 
        continuum fit and the integrated feature regions.
    definitions : dict
        A dictionary containing index definitions. 
        Format: {'IndexName': {'unit': 'A' or 'mag', 
                               'continuum': [[s1, e1], [s2, e2]], 
                               'feature': [[s3, e3]]}}

    Returns:
    --------
    EW : float
        The measured index value. For unit='A', this is the Equivalent Width. 
        For unit='mag', this is the Lick-style magnitude.
    EWe : float (only if err is provided)
        The 1-sigma uncertainty of the measurement, propagated through 
        the trapezoidal integration.

    Methodology:
    ------------
    1. Slices the spectrum to the region of interest plus 20% padding.
    2. Calculates 'anchor points' by averaging flux in continuum windows.
    3. Fits a polynomial of degree 'coeff' to these anchor points.
    4. Normalizes the flux (and errors) by this modeled continuum.
    5. Interpolates the exact boundaries of feature windows for precision.
    6. Integrates using the trapezoidal rule.
       - Unit 'A': sum(delta_lambda * (1 - norm_flux))
       - Unit 'mag': -2.5 * log10(sum(flux_integral) / sum(window_widths))
    """
    if definitions == None:
         raise KeyError('Please provide definitions for the indices.')
    if len(wl) != len(flx):
         raise ValueError(f'Wavelength and flux arrays must be the same length\n'+
                          f'len(wl) = {len(wl)}, and len(flx) = {len(flx)}.')
    if ind is None:
         raise TypeError('You must provide an index to measure')

    if (err is not None) and (len(wl) != len(err)):
         raise ValueError(f'Wavelength, flux and error arrays must be the same length\n'+
                          f'len(wl) = {len(wl)}, len(flx) = {len(flx)}, len(err) = {len(err)}.')

    try:
         wl  = np.array(wl)
         flx = np.array(flx)
         if err is not None:
              err = np.array(err)
    except:
         raise ValueError(f'One of your objects (wl, flx, err) could not be converted to a numpy array.')

    if not np.all(np.diff(wl) > 0):
        raise ValueError('Wavelength array must be strictly increasing. '
                         'Check for unsorted data or duplicate values.')

    
    if ind not in definitions:
        raise KeyError(f"Index '{ind}' not found in definitions file.")
    
    line_def = definitions[ind]

    required_keys = ['unit', 'continuum', 'feature']
    if not all(key in line_def for key in required_keys):
        missing = [k for k in required_keys if k not in line_def]
        raise KeyError(f"Index '{ind}' definition is missing: {', '.join(missing)}")
    
    unit         = line_def['unit'] # Default to Angstroms if not specified
    cont_windows = line_def['continuum']

    if len(cont_windows) < 2:
         raise ValueError(f'Index {ind} has less than two bandpasses: {cont_windows}')
    
    feat_windows = line_def['feature']

    all_boundaries = [val for window in (cont_windows + feat_windows) for val in window]

    index_min = min(all_boundaries)
    index_max = max(all_boundaries)

    if (index_min < min(wl)) or  (index_max > max(wl)):
         raise ValueError(f'Wavelength covers between {min(wl)} and {max(wl)}, but your\n'+
                          f'index definitions go from {index_min} to {index_max}.')

         
    padding = (index_max - index_min)*0.2
    mask = (wl>index_min-padding) & (wl<index_max+padding)
    wl  = wl[mask]
    flx = flx[mask]
    if err is not None:
         err = err[mask]


    # Processing the bandpasses
    bandpasses_x = []
    bandpasses_y = []
    for cont_window in cont_windows:
         if len(cont_window) != 2:
              raise ValueError(f'Bandpass {cont_window} has length {len(cont_window)}.')
         bandpasses_x.append(np.mean(cont_window))
         bandpasses_y.append(np.nanmean(flx[(wl>cont_window[0]) & (wl<cont_window[1])]))

    bandpasses_x = np.array(bandpasses_x)
    bandpasses_y = np.array(bandpasses_y)

    #Fitting the continuum to normalize the bandpasses
    coeffs = np.polyfit(bandpasses_x, bandpasses_y, coeff)
    continuum_flux = np.polyval(coeffs, wl)
    
    if plot:
        f, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15, 8))
        ax[0].plot(wl, flx, ls = '-', color = 'k', alpha = 0.5)
        ax[0].plot(wl, continuum_flux, ls = '--', color = 'k')
        for cont_window in cont_windows:
             pltmask = (wl>cont_window[0]) & (wl<cont_window[1])
             ax[0].plot(wl[pltmask], flx[pltmask], ls = '-', color = 'r')
        for feat_window in feat_windows:
             ax[0].axvspan(feat_window[0], feat_window[1], color = 'b', alpha = 0.5)
        for i in range(len(bandpasses_x)):
             ax[0].plot(bandpasses_x[i], bandpasses_y[i], color = 'r', ls = None, marker = 'x', markersize = 15)

    flx = flx/continuum_flux
    if err is not None:
         err = err/continuum_flux
         var_total_int = 0 # For error propagation

    if plot:
        ax[1].plot(wl, flx,    ls = '-',  color = 'k', alpha = 0.5)
        ax[1].plot(wl, wl*0+1, ls = '--', color = 'k')
        for cont_window in cont_windows:
             pltmask = (wl>cont_window[0]) & (wl<cont_window[1])
             ax[1].plot(wl[pltmask], flx[pltmask], ls = '-', color = 'r')

    #finally, integrate to get the feature
    total_flux_int = 0
    total_width = 0

    for feat_window in feat_windows:
        featmask = (wl>feat_window[0]) & (wl<feat_window[1])
        w_feat = np.concatenate([
           [feat_window[0]],
           wl[featmask],
           [feat_window[1]]   ])
        f_feat = np.concatenate([
           [np.interp(feat_window[0], wl, flx)],
           flx[featmask],
           [np.interp(feat_window[1], wl, flx)]  ])
        total_flux_int += np.trapz(f_feat, w_feat)
        total_width += (feat_window[1] - feat_window[0])
        if err is not None:
             e_feat = np.concatenate([
                  [np.interp(feat_window[0], wl, err)],
                  err[featmask],
                  [np.interp(feat_window[1], wl, err)]
             ])
             dw  = np.diff(w_feat)
             var_total_int += np.sum((dw**2) * (e_feat[:-1]**2 + e_feat[1:]**2) / 4)
             
        if plot:
             ax[1].fill_between(w_feat, f_feat, 1, color = 'b', alpha = 0.5)

    if unit == 'A':
        EW = total_width - total_flux_int
    elif unit == 'mag':
        mean_flux_ratio = total_flux_int / total_width
        mean_flux_ratio = np.maximum(mean_flux_ratio, 1e-9) # Safety clamp
        EW = -2.5 * np.log10(mean_flux_ratio)

    if err is not None:
        if unit == 'A':
            EWe = np.sqrt(var_total_int)
        elif unit == 'mag':
            EWe = (2.5 / np.log(10)) * (np.sqrt(var_total_int) / total_flux_int)

    if plot:#
         true_unit = '$\\AA$' if unit == 'A' else unit
         print(f'{ind} = {round(EW,3)} {true_unit}')
         if err is not None:
              ax[0].set_title(f'{ind} = {round(EW,2)}$\pm${round(EWe,2)} {true_unit}')
         else:
              ax[0].set_title(f'{ind} = {round(EW,3)} {true_unit}')
         f.subplots_adjust(hspace=0.)
         ax[0].set_xticks([])
         ax[1].set_xlabel('Wavelength ($\\AA$)')
         ax[0].set_xlim(index_min-padding, index_max+padding)
         ax[1].set_xlim(index_min-padding, index_max+padding)
         ax[0].set_ylabel('Flux')
         ax[1].set_ylabel('Norm. Flux')
         plt.show()

    if err is not None:
         return(EW, EWe)
    else:
         return(EW)


