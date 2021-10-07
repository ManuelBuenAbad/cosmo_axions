#######################################################
###    Code for probabilities for axion-photon      ###
###              conversion in the ICM              ###
###          by Manuel A. Buen-Abad, 2020           ###
###               and Chen Sun, 2020                ###
#######################################################

from __future__ import division

import numpy as np

from numpy import pi, sqrt, log, log10, exp, power, cumprod
from scipy.integrate import simps, quad
from scipy.interpolate import interp1d
from inspect import getargspec
from ag_probs import omega_plasma, P0


# FUNCTIONS:


def L_dist(L, L_low=3.5, L_up=10., n=-1.2):
    """
    Power law distribution of magnetic domain sizes.

    L : domain size [kpc]
    L_low : domain size lower bound [kpc] (default: 3.5)
    L_up : domain size upper bound [kpc] (default: 10.)
    n : power law (default: -1.2)
    """

    normal = (L_up**(n+1.))/(1+n) - (L_low**(n+1.))/(1+n)

    return L**n / normal



L_avg = quad(lambda l: L_dist(l) * l, 3.5, 10.)[0]



def ne_2beta(r, ne0=0.01, rc_outer=100., beta_outer=1., f_inner=0., rc_inner=10., beta_inner=1.):
    """
    Electron number density [cm^-3] in the double-beta profile of the hydrostratic equilibrium model.

    r : distance from the center of the cluster [kpc]
    ne0 : central electron number density [cm^-3]
    rc_outer : core radius from the outer component [kpc] (default: 100.)
    beta_outer : slope from the outer component (default: 1.)
    f_inner : fractional contribution from inner component (default: 0.)
    rc_inner : core radius from the inner component [kpc] (default: 10.)
    beta_inner : slope from the inner component (default: 1.)
    """

    outer = lambda rr: (1. + rr**2./rc_outer**2.)**(-1.5*beta_outer) # outer contribution
    inner = lambda rr: (1. + rr**2./rc_inner**2.)**(-1.5*beta_inner) # inner contribution

    return ne0*( f_inner*inner(r) + (1.-f_inner)*outer(r) )



def B_icm(r, ne_fn, B_ref=10., r_ref=0., eta=0.5, **kwargs):
    """
    Magnetic field [muG] in the ICM, proportional to a power of the electron number density.

    r : distance from the center of the cluster [kpc]
    ne_fn : function for the electron number density [cm^-3]
    B_ref : reference value of the magnetic field [muG] (default: 10.)
    r_ref : reference value of the radius [kpc] (default: 0.)
    eta : power law of B_icm as a function of ne (default: 0.5)
    kwargs : other keyword arguments of the function 'ne_fn'
    """

    return B_ref*(ne_fn(r, **kwargs)/ne_fn(r_ref, **kwargs))**eta



def icm_Psurv(ma, g, r_ini, r_fin, ne_fn, B_fn,
              L=10.,
              omega_Xrays=10.,
              axion_ini_frac=0.,
              smoothed=False,
              method='product',
              # if method=='product':
              return_arrays=False,
              # if method=='simps'/'quad':
              prob_func='norm_log',
              # if method=='simps':
              Nr=501,
              mu=1.,
              **kwargs):
    """
    ICM survival probability for photons originating at a distance r from the cluster's center.

    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    r_ini : photon initial radial distance to the cluster center [kpc]
    r_fin : photon final radial distance to the cluster center [kpc]
    ne_fn : function for the electron number density [cm^-3]
    B_fn : function for the ICM magnetic field [muG]
    L : ICM magnetic field domain size [kpc] (default: 10.)
    omega_Xrays : photon energy [keV] (default: 10.)
    axion_ini_frac : the initial intensity fraction of axions: I_axion/I_photon (default: 0.)
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    method : the integration method 'simps'/'quad'/'product' (default: 'product')

    # if method=='product':
    return_arrays : whether we return the partial products and radii arrays (useful for icm_los_Psurv) [bool] (default: False)

    # if method=='simps'/'quad':
    prob_func : the form of the probability function: 'small_P' for the P<<1 limit, 'full_log' for log(1-1.5*P), and 'norm_log' for the normalized log: log(abs(1-1.5*P)) [str] (default: 'norm_log')

    # if method=='simps':
    Nr : number of radius bins (default: 501)

    mu : signal strength (default: 1.)

    kwargs : other keyword arguments of the functions 'ne_fn' and 'B_fn'
    """

    if (return_arrays and (method != 'product')):
        raise ValueError("If you use return_arrays = True you need method='product'.")

    A = (2./3)*(1 + axion_ini_frac) # equilibration constant

    # reading the parameter names of ne_fn and B_fn
    ne_pars = getargspec(ne_fn)[0]
    B_pars = getargspec(B_fn)[0]

    # building the kwargs for ne_fn and B_fn
    ne_kwargs = {}
    B_kwargs = {}
    for key, val in kwargs.items():
        if key in ne_pars:
            ne_kwargs[key] = val
        if key in B_pars:
            B_kwargs[key] = val

    # defining functions of r
    ne = lambda rr: ne_fn(rr, **ne_kwargs) # ICM electron number density [cm^-3]
    mg = lambda rr: omega_plasma(ne(rr)) # photon plasma mass [eV]
    Bicm = lambda rr: B_fn(rr, ne_fn, **kwargs) # ICM magnetic field [muG]

    P = lambda rr: mu*P0(ma, g, L/1000., B=Bicm(rr)*1000., omega=omega_Xrays*1000., mg=mg(rr), smoothed=smoothed) # conversion probability in domain located at radius rr from center of cluster

    if method == 'product':

        N = int(round((r_fin - r_ini)/L)) # number of magnetic domains
        r_Arr = (r_ini + L/2.) + L*np.arange(N) # array of r-values of the domains' centers
        P_Arr = P(r_Arr) # array of conversion probabilities
        factors = 1. - 1.5*P_Arr # the factors in each domain

        total_prod = factors.prod()
        partial_prods = cumprod(factors[::-1])[::-1]

        if return_arrays: # we are asked to return the arrays of partial products and radii for later use

            return (A + (1.-A)*total_prod, A + (1.-A)*partial_prods, r_Arr)

        else: # we are asked to simply give the survival probability and nothing else

            return A + (1.-A)*total_prod

    elif method == 'simps':

        rArr = np.linspace(r_ini, r_fin, Nr)

        if prob_func == 'norm_log':
            integrand = log( np.abs(1. - 1.5*P(rArr)) )
        elif prob_func == 'small_P':
            integrand = -1.5*P(rArr)
        elif prob_func == 'full_log':
            integrand = log( 1. - 1.5*P(rArr) )
        else:
            raise ValueError("Argument 'prob_func'={} must be equal to either 'small_P', 'full_log', or 'norm_log'. It's neither.".format(prob_func))

        integral = simps(integrand, rArr)
        argument = integral/L

        return A + (1.-A)*exp(argument)

    elif method == 'quad':

        if prob_func == 'norm_log':
            integrand = lambda rr: log( np.abs(1. - 1.5*P(rr)) )
        elif prob_func == 'small_P':
            integrand = lambda rr: -1.5*P(rr)
        elif prob_func == 'full_log':
            integrand = lambda rr: log( 1. - 1.5*P(rr) )
        else:
            raise ValueError("Argument 'prob_func'={} must be equal to either 'small_P', 'full_log', or 'norm_log'. It's neither.".format(prob_func))

        integral = quad(integrand, r_ini, r_fin)[0]
        argument = integral/L

        Pconv = (1.-A)*(1.-exp(argument)) # conversion probability
        Psurv = 1. - Pconv # survival probability
        # A + (1.-A)*exp(argument) # old return

        return Psurv

    else:
        raise ValueError("Argument 'method'={} must be equal to either 'simps', 'quad', or 'product'. It's neither.".format(method))



def icm_los_Psurv(ma, g, r_low, r_up, ne_fn, B_fn,
                  L=10.,
                  omega_Xrays=10.,
                  axion_ini_frac=0.,
                  smoothed=False,
                  method='product',
                  # if method=='product':
                  return_arrays=False,
                  # if method=='simps'/'quad':
                  prob_func='norm_log',
                  # if method=='simps':
                  Nr=501,
                  # for l.o.s. integration:
                  los_method='quad',
                  # if los_method=='simps' && method=='product' && return_arrays=True:
                  los_use_prepared_arrays=False,
                  # if los_method=='simps' && los_use_prepared_arrays=False:
                  los_Nr=501,
                  mu = 1.,
                  **kwargs):
    """
    Line-of-sight average of the photons ICM survival probability.

    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    r_low : lower end of the integration [kpc]
    r_up : upper end of the integration [kpc]
    ne_fn : function for the electron number density [cm^-3]
    B_fn : function for the ICM magnetic field [muG]
    L : ICM magnetic field domain size [kpc] (default: 10.)
    omega_Xrays : photon energy [keV] (default: 10.)
    axion_ini_frac : the initial intensity fraction of axions: I_axion/I_photon (default: 0.)
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    method : the integration method 'simps'/'quad'/'product' (default: 'product')

    # if method=='product':
    return_arrays : whether we return the partial products and radii arrays (useful for icm_los_Psurv) [bool] (default: False)

    # if method=='simps'/'quad':
    prob_func : the form of the probability function: 'small_P' for the P<<1 limit, 'full_log' for log(1-1.5*P), and 'norm_log' for the normalized log: log(abs(1-1.5*P)) [str] (default: 'norm_log')

    # if method=='simps':
    Nr : number of radius bins, for the 'simps' methods (default: 501)

    # for l.o.s. integration:
    los_method : the integration method along the line of sight 'simps'/'quad' (default: 'simps')

    # if los_method=='simps' && method=='product' && return_arrays=True:
    los_use_prepared_arrays

    # if los_method=='simps' && los_use_prepared_arrays=False:
    los_Nr : number of radius bins along the line of sight, for the 'simps' methods (default: 501)

    mu : signal strength (default: 1.)

    kwargs : other keyword arguments of the functions 'ne_fn' and 'B_fn'
    """

    if los_use_prepared_arrays and (not return_arrays):
        raise ValueError("You cannot pass los_use_prepared_arrays=True if you have return_arrays=False. You cannot use arrays that aren't there!")

    # reading the parameter names of ne_fn and B_fn
    ne_pars = getargspec(ne_fn)[0]
    B_pars = getargspec(B_fn)[0]

    # building the kwargs for ne_fn and B_fn
    ne_kwargs = {}
    B_kwargs = {}
    for key, val in kwargs.items():
        if key in ne_pars:
            ne_kwargs[key] = val
        if key in B_pars:
            B_kwargs[key] = val

    # defining functions of r
    ne2 = lambda rr: ne_fn(rr, **ne_kwargs)**2. # square of the ICM electron number density [cm^-6]

    if return_arrays:

        _, pArr, rArr = icm_Psurv(ma, g, r_low, r_up, ne_fn, B_fn,
                                  L=L,
                                  omega_Xrays=omega_Xrays,
                                  axion_ini_frac=axion_ini_frac,
                                  smoothed=smoothed,
                                  method=method,
                                  return_arrays=return_arrays, # should be True
                                  prob_func=prob_func,
                                  Nr=Nr,
                                  mu=mu,
                                  **kwargs)

        pfn = interp1d(rArr, pArr, fill_value='extrapolate')
        Pgg_ne2 = lambda rr: ne2(rr) * pfn(rr)

    else:

        Pgg_ne2 = lambda rr: ne2(rr) * icm_Psurv(ma, g, rr, r_up, ne_fn, B_fn,
                                                 L=L,
                                                 omega_Xrays=omega_Xrays,
                                                 axion_ini_frac=axion_ini_frac,
                                                 smoothed=smoothed,
                                                 method=method,
                                                 return_arrays=return_arrays, # should be False
                                                 prob_func=prob_func,
                                                 Nr=Nr,
                                                 mu=mu,
                                                 **kwargs)

    if los_method == 'quad': # this method requires functions

        num = quad(Pgg_ne2, r_low, r_up)[0]
        den = quad(ne2, r_low, r_up)[0]

    elif los_method == 'simps': # this method requires arrays

        if los_use_prepared_arrays: # in this case we already have arrays prepared, and we will reuse them for the simps integration

            low_idx = np.abs(rArr - r_low).argmin() # finding the array index closest to the lower end of the l.o.s. integration
            up_idx = np.abs(rArr - r_up).argmin() # finding the array index closest to the upper end of the l.o.s. integration

            los_rArr = rArr[low_idx:up_idx+1] # the radii array
            ne2_Arr = ne2(los_rArr) # the ne2 array
            Pgg_ne2_Arr = ne2_Arr * pArr[low_idx:up_idx+1] # the ne2*Pgg array

            del low_idx, up_idx

        else: # we need to prepare the arrays for the simps integration

            los_rArr = np.linspace(r_low, r_up, los_Nr) # the radii array
            ne2_Arr = ne2(los_rArr) # the ne2 array

            Pgg_ne2_Arr = [] # the ne2*Pgg array

            for r in los_rArr:

                if not np.isnan(Pgg_ne2(r)):
                    Pgg_ne2_Arr.append(Pgg_ne2(r))
                else:
                    Pgg_ne2_Arr.append(0.)

            Pgg_ne2_Arr = np.array(Pgg_ne2_Arr)


        num = simps(Pgg_ne2_Arr, los_rArr)
        den = simps(ne2_Arr, los_rArr)

        del los_rArr, ne2_Arr, Pgg_ne2_Arr

    else:
        raise ValueError("Argument 'los_method'={} must be equal to either 'simps' or 'quad'. It's neither.".format(los_method))

    return num/den
