###########################################
###    Code for cosmology functions     ###
###    by Manuel A. Buen-Abad, 2020     ###
###         and Chen Sun, 2020          ###
###########################################

from __future__ import division

import numpy as np

from numpy import pi, sqrt, log, log10, exp, power
from scipy.integrate import simps, quad

from ag_probs import omega_plasma, P0, Pnaive
from igm import igm_Psurv
from icm import ne_2beta, B_icm, icm_los_Psurv


# CONSTANTS:

_c_ = 299792458.  # [m/s]
_alpha_ = 1./137  # fine structure constant
_me_ = 510998.95  # electron mass in eV
_1_over_cm_eV_ = 1.9732698045930252e-5  # [1/cm/eV]

huge = 1.e50 # a huge number
tiny = 1.e-50 # a tiny number

# FUNCTIONS:

def Ekernel(OmL, z):
    try:
        res, _ = quad( lambda zp: 1 / sqrt(OmL + (1 - OmL) * (1 + zp)**3), 0, z )
    except Warning:
        print('OmL=%e, z=%e' % (OmL, z))
        raise Exception
    return res



def H_at_z(z, h0, OmL, unit='Mpc'):
    """
    Hubble at z

    :param z: redshift
    :param h0:  H in [100*km/s/Mpc]
    :param OmL: Omega_Lambda
    :param unit: flag to change the output unit
    :returns: H [1/Mpc] by default, or H [km/s/Mpc]

    """
    if unit == 'Mpc':
        res = h0*100.*sqrt(OmL + (1 - OmL) * (1 + z)**3)/(_c_/1000.)
    else:
        res = h0*100.*sqrt(OmL + (1 - OmL) * (1 + z)**3)
    return res



def tau_at_z(z, h0, OmL):
    """
    Compute the comoving distance, return in Mpc

    Parameters
    ----------
    z : scalar
        redshift
    h0 : scalar
        Hubble in 100 km/s/Mpc
    OmL : scalar
        Omega_Lambda

    """
    try:
        res, _ = quad( lambda zp: 1. / sqrt(OmL + (1 - OmL) * (1 + zp)**3), 0., z )
    except Warning:
        print('OmL=%e, z=%e' % (OmL, z))
        raise Exception
    res = res * _c_/1e5/h0
    return res



def dA_at_z(z, h0, OmL):
    """
    Angular distance [Mpc]

    :param z: redshift
    :param h0: H in [100*km/s/Mpc]
    :param OmL: Omega_Lambda
    :returns: angular distance [Mpc]

    """
    return tau_at_z(z, h0, OmL)/(1.+z)



def muLCDM(z, h0, OmL):
    try:
        res = 5. * log10((1.+z) * Ekernel(OmL, z)) + 5.*log10(_c_/(h0*1e5)) + 25
    except Warning:
        print('z=%e, OmL=%e' % (z, OmL))
        print('h0=%e' % h0)
        print('(1+z)*Ekernel=%e, c/h0=%e' % ((1. + z) * Ekernel(OmL, z), _c_ / (h0 * 1e5)))
    return res



def LumMod(ma, g, z, B, mg, h, OmL,
           s=1.,
           omega=1.,
           axion_ini_frac=0.,
           smoothed=False,
           redshift_dependent=True,
           method='simps',
           prob_func='norm_log',
           Nz=501,
           mu=1.):
    """
    Here we use a simple function to modify the intrinsic luminosity of the SN
    so that mu_th = mu_STD - LumMod(). This is the one that takes into account the redshift

    Parameters
    ----------
    ma: axion mass [eV]
    g: axion photon coupling  [1/GeV]
    z: redshift
    B: magnetic field, today [nG]
    mg: photon mass [eV]
    h: Hubble [100 km/s/Mpc]
    OmL: Omega_Lambda
    s: domain size [Mpc]
    omega: energy [eV]

    Returns
    -------
    res: scalar, delta M in the note

    """

    try:
        # 2.5log10(L/L(1e-5Mpc))
        res = 2.5 * log10(igm_Psurv(ma, g, z,
                                    s=s,
                                    B=B,
                                    omega=omega,
                                    mg=mg,
                                    h=h,
                                    Omega_L=OmL,
                                    axion_ini_frac=axion_ini_frac,
                                    smoothed=smoothed,
                                    redshift_dependent=redshift_dependent,
                                    method=method,
                                    prob_func=prob_func,
                                    Nz=Nz,
                                    mu=mu))

    except Warning:
        print('ma=%e, g=%e, y=%e' % (ma, g, y))
        raise Exception('Overflow!!!')
    return res



def ADDMod(ma, g, z, h, OmL,

           omegaX=1.e4,
           omegaCMB=2.4e-4,

           # IGM
           sIGM=1.,
           BIGM=1.,
           mgIGM=3.e-15,
           smoothed_IGM=False,
           redshift_dependent=True,
           method_IGM='simps',
           prob_func_IGM='norm_log',
           Nz_IGM=501,

           # ICM
           ICM_effect=False,
           r_low = 0.,
           r_up = 1800.,
           L=10.,
           smoothed_ICM=False,
           method_ICM='product',
           return_arrays=False,
           prob_func_ICM='norm_log',
           Nr_ICM=501,
           los_method='quad',
           los_use_prepared_arrays=False,
           los_Nr=501,

           mu=1.,

           # B_icm
           B_ref=10.,
           r_ref=0.,
           eta=0.5,

           #ne_2beta
           ne0=0.01,
           rc_outer=100.,
           beta_outer=1.,
           f_inner=0.,
           rc_inner=10.,
           beta_inner=1.):
    """
    Function that modifies the ADDs from clusters, written in Eq. 12 of Manuel's notes.
    """

    if ICM_effect:

        PICM_X = icm_los_Psurv(ma, g, r_low, r_up, ne_2beta, B_icm,
                               L=L,
                               omega_Xrays=omegaX/1000.,
                               axion_ini_frac=0.,
                               smoothed=smoothed_ICM, method=method_ICM, return_arrays=return_arrays, prob_func=prob_func_ICM, Nr=Nr_ICM, los_method=los_method, los_use_prepared_arrays=los_use_prepared_arrays, los_Nr=los_Nr,
                               mu=mu,
                               # B_icm
                               B_ref=B_ref, r_ref=r_ref, eta=eta,
                               # ne_2beta
                               ne0=ne0, rc_outer=rc_outer, beta_outer=beta_outer, f_inner=f_inner, rc_inner=rc_inner, beta_inner=beta_inner)
        
        PICM_SZ = icm_los_Psurv(ma, g, r_low, r_up, ne_2beta, B_icm,
                                L=L,
                                omega_Xrays=omegaCMB/1000.,
                                axion_ini_frac=0.,
                                smoothed=smoothed_ICM, method=method_ICM, return_arrays=return_arrays, prob_func=prob_func_ICM, Nr=Nr_ICM, los_method=los_method, los_use_prepared_arrays=los_use_prepared_arrays, los_Nr=los_Nr,
                                mu=mu,
                                # B_icm
                                B_ref=B_ref, r_ref=r_ref, eta=eta,
                                # ne_2beta
                                ne0=ne0, rc_outer=rc_outer, beta_outer=beta_outer, f_inner=f_inner, rc_inner=rc_inner, beta_inner=beta_inner)

        # regularizing the SZ and X-ray ICM survival probabilities:
        PICM_X = np.clip(PICM_X, -huge, huge)
        PICM_SZ = np.clip(PICM_SZ, -huge, huge)
        
        PgX, PaX = PICM_X, 1.-PICM_X
        PgSZ, PaSZ = PICM_SZ, 1.-PICM_SZ
        
        IaIgX = PaX/PgX
        IaIgSZ = PaSZ/PgSZ

    else:
        PgX, PgSZ = 1., 1.
        IaIgX, IaIgSZ = 0., 0.

    PIGM_X = igm_Psurv(ma, g, z,
                       s=sIGM,
                       B=BIGM,
                       omega=omegaX,
                       mg=mgIGM,
                       h=h,
                       Omega_L=OmL,
                       axion_ini_frac=IaIgX,
                       smoothed=smoothed_IGM,
                       redshift_dependent=redshift_dependent,
                       method=method_IGM,
                       prob_func=prob_func_IGM,
                       Nz=Nz_IGM,
                       mu=mu)

    PIGM_SZ = igm_Psurv(ma, g, z,
                        s=sIGM,
                        B=BIGM,
                        omega=omegaCMB,
                        mg=mgIGM,
                        h=h,
                        Omega_L=OmL,
                        axion_ini_frac=IaIgSZ,
                        smoothed=smoothed_IGM,
                        redshift_dependent=redshift_dependent,
                        method=method_IGM,
                        prob_func=prob_func_IGM,
                        Nz=Nz_IGM,
                        mu=mu)

    print "IGM_SZ={:.2e}, IGM_X={:.2e}\tICM_SZ={:.2e}, ICM_X={:.2e}".format(PIGM_SZ, PIGM_X, PgSZ, PgX)
    
    # regularizing the CMB and X-ray IGM survival probabilities
    PIGM_SZ = np.clip(PIGM_SZ, -huge, huge)
    PIGM_X = np.clip(PIGM_X, -huge, huge)
    
    modif = (PIGM_SZ**2. * PgSZ**2.) / (PIGM_X * PgX)
    # regularizing again:
    modif = np.clip(modif, -huge, huge)
    
    return modif
