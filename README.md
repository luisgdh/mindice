# mindice
Measure spectral indices (Equivalent Width or Magnitudes) for stellar/galactic spectra.

This function performs a local continuum normalization using a polynomial fit 
across multiple user-defined bandpasses and integrates the flux within 
specified feature windows. It supports both Angstrom (EW) and Magnitude 
(Lick-style) units, handling multiple feature bands for a single index.

## Installation
In order to install, run the following in your terminal:
```
$ git clone https://github.com/luisgdh/mindice/
$ cd mindice
$ pip install -e .
```

## How to run:
```
from mindice import mindice
mindice(wl, flx, err=None, ind=None, coeff=1, plot=False, definitions=None)
```

## Example:
```
spec    = np.loadtxt('NGC1052.spec', unpack = True)
mindice(spec[0], spec[1], ind = 'Hb', plot = True)
```

### Parameters:
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