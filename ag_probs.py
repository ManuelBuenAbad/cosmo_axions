################################################################
###    Code for probabilities for axion-photon conversion    ###
###              by Manuel A. Buen-Abad, 2020                ###
###                  and Chen Sun, 2020                      ###
################################################################

from __future__ import division

import numpy as np

from numpy import pi, sqrt, log, log10, exp, power
from scipy.integrate import simps, quad

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

    x_ieV = x*Mpc_times_eV

    pref = (Delta(g, B=B)**2.) / (k(ma, g, B=B, omega=omega, mg=mg)**2.)
    arg = (k(ma, g, B=B, omega=omega, mg=mg) * x_ieV)/2.

    if not smoothed:
        try:
            osc = np.sin(arg)**2
        except:
            try:
                osc = (1 - exp(-2*arg**2.))/2.
            except:
                osc = -huge
    else:
        try:
            osc = (1 - exp(-2*arg**2.))/2.
        except:
            osc = -huge

    return np.real(pref * osc)

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

    return A + (1-A)*exp(argument)
