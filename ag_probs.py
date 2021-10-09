################################################################
###    Code for probabilities for axion-photon conversion    ###
###              by Manuel A. Buen-Abad, 2020                ###
###                  and Chen Sun, 2020                      ###
################################################################

from __future__ import division

import numpy as np

from numpy import pi, sqrt, log, log10, exp, power
from scipy.integrate import simps, quad

# import warnings

# # In order to handle RuntimeWarning overflows;
# warnings.filterwarnings("error")


# CONSTANTS AND CONVERSION FACTORS:

c0 = 299792458. # [m/s] speed of light
aem = 1./137 # QED coupling constant
meeV = (0.51099895 * 1.e6) # [eV] electron mass
hbarc = 197.32698045930252e-18 # [GeV*cm]
GeV_over_eV = 1.e9 # GeV/eV

GeV_times_m = 1./hbarc # GeV*m conversion
eV_times_cm = GeV_times_m * 1.e-11 # eV*cm conversion

Mpc_over_m = 3.085677581282e22 # Mpc/m conversion
Mpc_times_GeV = GeV_times_m * Mpc_over_m # Mpc*GeV conversion
G_over_eV2 = 1.95e-2 # G/eV^2 conversion # MANUEL: NOTE: FIND MORE ACCURATE VALUE

Mpc_times_eV = Mpc_times_GeV/GeV_over_eV # Mpc*eV conversion

huge = 1.e50 # huge number
tiny = 1.e-15 # tiny number

# FUNCTIONS:

def omega_plasma(ne):
    """
    Plasma photon mass [eV].

    ne : electron number density [cm^-3]
    """

    ne_eV3 = ne*(eV_times_cm)**-3

    omega2 = 4*pi*aem*ne_eV3 / meeV

    return sqrt(omega2)


def Delta(g, B=1.):
    """
    Delta parameter.

    g : axion-photon coupling [GeV^-2]
    B : magnetic field [nG] (default: 1.)
    """
    g_ieV = (g/GeV_over_eV)
    B_eV2 = (B*1.e-9) * G_over_eV2

    return g_ieV*B_eV2

def q(ma, omega=1., mg=3.e-15):
    """
    q parameter.

    ma : axion mass [eV]
    omega : photon energy [eV] (default: 1.)
    mg : photon mass [eV] (default: 3.e-15)
    """

    return (mg**2)*(1. - power(ma/mg, 2.))/omega

def k(ma, g, B=1., omega=1., mg=3.e-15):
    """
    Oscillation wavenumber.

    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    B : magnetic field [nG] (default: 1.)
    omega : photon energy [eV] (default: 1.)
    mg : photon mass [eV] (default: 3.e-15)
    """

    return sqrt( Delta(g, B=B)**2. + (q(ma, omega=omega, mg=mg)/2.)**2. )


def treat_as_arr(arg):
    """
    A routine to cleverly return scalars as (temporary and fake) arrays. True arrays are returned unharmed. Thanks to Chen!
    """
    
    arr = np.asarray(arg)
    is_scalar = False
    
    # making sure scalars are treated properly
    if arr.ndim == 0: # it is really a scalar!
        arr = arr[None] # turning scalar into temporary fake array
        is_scalar = True # keeping track of its scalar nature
    
    return arr, is_scalar


def reg_arr(arr, lo_real_cutoff=-huge, hi_real_cutoff=huge, lo_imag_cutoff=-huge, hi_imag_cutoff=huge):
    """
    A routine to reguarize a complex array.
    """
    
    # real...
    real_arr = np.real(arr)
    # ... and imaginary parts (if present)
    imag_arr = np.imag(arr)
    
    # regularizing arg_arr:
    # the real part...
    real_arr = np.clip(real_arr, lo_real_cutoff, hi_real_cutoff)
    
    if np.any(imag_arr):
        # ... and the imaginary part (if present)
        imag_arr = np.clip(imag_arr, lo_imag_cutoff, hi_imag_cutoff)
    
#     # first the real part...
#     arr = np.where(np.abs(np.real(arr)) > real_cutoff,
#                    np.sign(np.real(arr))*real_val + np.imag(arr)*1j,
#                    arr)
#     # then the imaginary part...
#     arr = np.where(np.abs(np.imag(arr)) > imag_cutoff,
#                    np.real(arr) + np.sign(np.imag(arr))*imag_val*1j,
#                    arr)

    if np.any(imag_arr):
        return real_arr + imag_arr*1j
    else:
        return real_arr
    

def osc_fn(arg, smoothed=False):
    """
    A function describing the oscillation pattern of the probabilities.
    
    arg : argument of the oscillation function. Each entry should be either purely real or purely imaginary.
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    """
    
    # clever scalar -> array trick
    arr, is_scalar = treat_as_arr(arg)
    
    # Testing whether arg has elements that are have both real and imaginary parts (i.e. complex).
    real_abs = np.abs(np.real(arr))
    imag_abs = np.abs(np.imag(arr))
    # clipping possible error-size imaginary parts (should never happen!)
    imag_abs = np.where(imag_abs < tiny, 0., imag_abs)
    # whether there is at least an element with both real and imaginary parts:
    is_complex = np.any(real_abs*imag_abs)
    
    if is_complex:
        raise ValueError("'arg' is complex: neither purely real, nor purely imaginary. Cannot currently handle this.")
    
    # preparing array of oscillation values:
    osc = np.ones_like(arr)
    
    # computing the oscillatory part:
    if (np.any(np.imag(arr)) or (not smoothed)):
        
        # regularizing arr. Note that we're less forgiving with imaginary parts: that's because they will yield exponentials
        reg = reg_arr(arr,
                      lo_real_cutoff=-huge,
                      hi_real_cutoff=huge,
                      lo_imag_cutoff=-log10(huge),
                      hi_imag_cutoff=log10(huge))
        
        osc = np.sin(reg)**2
        
    else:
        # regularizing arr. Note that the imaginary cutoffs should be irrelevant, since the logic precludes the use of the smoothed function in the case where arr contains imaginary elements
        reg = reg_arr(arr,
                      lo_real_cutoff=-huge,
                      hi_real_cutoff=huge,
                      lo_imag_cutoff=-log10(huge),
                      hi_imag_cutoff=log10(huge))
        
        osc = (1 - exp(-2*reg**2.))/2.
    
    if is_scalar: # returning fake arrays to scalar form
        osc = np.squeeze(osc)
    
    return osc


def P0(ma, g, x, B=1., omega=1., mg=3.e-15, smoothed=False):
    """
    Probability of axion-photon conversion in uniform magnetic field.

    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    x : distance traveled [Mpc]
    B : magnetic field [nG] (default: 1.)
    omega : photon energy [eV] (default: 1.)
    mg : photon mass [eV] (default: 3.e-15)
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    """
    
    x_ieV = x*Mpc_times_eV # [eV]
    
    # argument of oscillatory function:
    arg = (k(ma, g, B=B, omega=omega, mg=mg) * x_ieV)/2.
    
    # prefactor of probability conversion:
    pref = (Delta(g, B=B)**2.) / (k(ma, g, B=B, omega=omega, mg=mg)**2.)
    
    # clever scalar -> array trick
    arg_arr, is_scalar = treat_as_arr(arg)
    osc = osc_fn(arg_arr, smoothed=smoothed)
    
    # linear limit: abs(k*x/2) << 1 [sign of k^2 is canceled]
    lin_limit = Delta(g, B=B)**2. * x_ieV**2. / 4.
    
    # looking for where the linear regime is aplicable
    lin_here = np.where(np.abs(arg_arr) < tiny)[0]
    # and where we can use the full expression
    full_here = np.where(np.abs(arg_arr) >= tiny)[0]
    
#     print lin_here, full_here
    
#     print lin_limit
    
    # preparing the array of probabilities (extend to complex plane to allow for enough memory)
    res = np.ones_like(osc) + 0j
    # assigning the linear value...
    res[lin_here] = lin_limit[lin_here]
    # and the full value
    res[full_here] = (pref*osc)[full_here]
    
    if is_scalar:
        res = np.squeeze(res)
    
    if np.any(np.imag(res)):
        raise ValueError("There are still leftover imaginary parts in the probability. This should not have happened.")

    return np.real(res)

def Pnaive(ma, g, y, s=1., B=1., omega=1., mg=3.e-15, axion_ini_frac=0., smoothed=False):
    """
    Photon survival probability, in its naive (Grossman-Roy-Zupan) form.

    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    y : comoving distance traveled by the photons [Mpc]
    s : magnetic domain size [Mpc] (default: 1.)
    B : magnetic field [nG] (default: 1.)
    omega : photon energy [eV] (default: 1.)
    mg : photon mass [eV] (default: 3.e-15)
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    """

    A = (2./3)*(1 + axion_ini_frac) # equilibration constant
    N = y/s # number of magnetic domains
    P = P0(ma, g, s, B=B, omega=omega, mg=mg, smoothed=smoothed)
    argument = -1.5*N*P
    
    # regularizing the argument
    argument = np.clip(argument, -huge, log(huge))

    return A + (1-A)*exp(argument)
