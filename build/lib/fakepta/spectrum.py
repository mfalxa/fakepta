import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d

def powerlaw(f, log10_A, gamma):

    fyr = 1/sc.Julian_year
    psd_rn = (10**log10_A)** 2 / (12.0 * np.pi**2) * fyr**(gamma-3) * f**(-gamma)
    return psd_rn