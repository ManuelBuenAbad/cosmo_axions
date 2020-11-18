#######################################################
###           Code for chi2 calculations            ###
###          by Manuel A. Buen-Abad, 2020           ###
###               and Chen Sun, 2020                ###
#######################################################

import numpy as np
import scipy.linalg as la
from numpy import pi, sqrt, log, log10, exp, power
from cosmo import H_at_z, tau_at_z, dA_at_z, muLCDM, LumMod, ADDMod
import data


##########################
# auxiliary functions
##########################


def is_Out_of_Range(x, keys, params):
    """
    Returns a Boolean type indicating whether the current
    point is within the range

    Parameters
    ----------
    x : tuple
        the current point in the hyperspace to be checked
    keys: list
        each correspond to a dimension in the hyperspace,
        i.e. all the variables to be scanned
    """
    res = False

    for i in range(len(x)):
        if x[i] > params[keys[i]+' up'] or x[i] < params[keys[i]+' low']:
            res = True
            break
    return res



##########################
# chi2 functions
##########################



def chi2_SH0ES(M0, data=None):
    """
    Computes SH0ES chi2. data must be equal to (Anchor_SN, Anchor_SNsig, Anchor_Ceph, Anchor_Cephsig, Anchor_M, Anchor_Msig, aB, aBsig)
    """
    
    Anchor_SN, _, Anchor_Ceph, _, _, Anchor_Msig, _, _ = data
    
    chi2 = 0.
    
    for i in range(len(Anchor_SN)):
        chi2 += (Anchor_SN[i] - M0 - Anchor_Ceph[i])**2 / Anchor_Msig[i]**2
    
    return chi2



def chi2_BOSSDR12(x, data=None):
    """
    Computes BOSSDR12 chi2. data must be equal to (BOSS_rsfid, BOSS_meas_z, BOSS_meas_dM, BOSS_meas_Hz, BOSS_cov, BOSS_icov)
    """
    
    (OmL, h0, rs) = x
    BOSS_rsfid, BOSS_meas_z, BOSS_meas_dM, BOSS_meas_Hz, _, BOSS_icov = data
    
    chi2 = 0.
    data_array = np.array([], 'float64')
    
    for i, z in enumerate(BOSS_meas_z):
        
        DM_at_z = tau_at_z(z, h0, OmL)  # comoving
        H_at_z_val = H_at_z(z, h0, OmL, unit='SI')  # in km/s/Mpc
        
        theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rs * BOSS_rsfid
        theo_H_rd_by_rdfid = H_at_z_val * rs / BOSS_rsfid
        
        # calculate difference between the sampled point and observations
        DM_diff = theo_DM_rdfid_by_rd_in_Mpc - BOSS_meas_dM[i]
        H_diff = theo_H_rd_by_rdfid - BOSS_meas_Hz[i]
        
        # save to data array
        data_array = np.append(data_array, DM_diff)
        data_array = np.append(data_array, H_diff)
    
    chi2 += np.dot(np.dot(data_array, BOSS_icov), data_array)
    
    return chi2



def chi2_BAOlowz(x, data=None):
    """
    Computes BAOlowz chi2. data must be equal to (BAOlowz_meas_exp, BAOlowz_meas_z, BAOlowz_meas_rs_dV, BAOlowz_meas_sigma, BAOlowz_meas_type)
    """
    
    (OmL, h0, rs) = x
    _, BAOlowz_meas_z, BAOlowz_meas_rs_dV, BAOlowz_meas_sigma, BAOlowz_meas_type = data
    
    chi2 = 0.
    for i, z in enumerate(BAOlowz_meas_z):
        da = dA_at_z(z, h0, OmL)
        dr = z / H_at_z(z, h0, OmL)
        dv = (da * da * (1 + z) * (1 + z) * dr)**(1. / 3.)
        
        if BAOlowz_meas_type[i] == 3:
            theo = dv / rs
        elif BAOlowz_meas_type[i] == 7:
            theo = rs / dv
        chi2 += ((theo - BAOlowz_meas_rs_dV[i]) / BAOlowz_meas_sigma[i]) ** 2
    
    return chi2



def chi2_Pantheon(x, data=None, **kwargs):
    """
    Computes Pantheon chi2. data must be equal to (PAN_lkl, PAN_cov). **kwargs are the arguments for LumMod.
    """
    
    (ma, ga, OmL, h0, M0) = x
    PAN_lkl, PAN_cov = data
    
    chi2 = 0.
    residuals = []
    
    # numerical scan
    # analytically integrating out
    
    for rec in PAN_lkl:
        z = rec[0]
        m_meas = rec[1]
        
        change = LumMod(ma, ga, z, h=h0, OmL=OmL, **kwargs)
        
        residuals.append(muLCDM(z, h0, OmL) - m_meas + M0 - change)
        
    L_residuals = la.solve_triangular(PAN_cov, residuals, lower=True, check_finite=False)
    chi2 = np.dot(L_residuals, L_residuals)
    
    return chi2



def chi2_External(h0, data=None):
    """
    Computes h0 chi2. data must be equal to (h_TD, h_TD_sig).
    """
    h0_prior_mean, h0_prior_sig = data
    
    chi2 = 0.

    # add a Gaussian prior to H0
    
    chi2 += (h0 - h0_prior_mean)**2 / h0_prior_sig**2
    
    return chi2



def chi2_early(rs, data=None):
    """
    Computes rs chi2. data must be equal to (rsdrag_mean, rsdrag_sig).
    """
    
    rsdrag_prior_mean, rsdrag_prior_sig = data
    
    chi2 = 0.
    
    # add a Gaussian prior to rs
    
    chi2 += (rs - rsdrag_prior_mean)**2 / rsdrag_prior_sig**2
    
    return chi2



def chi2_clusters(pars, data=None, wanna_correct=True, fixed_Rvir=False, **kwargs):
    """
    Computes clusters chi2. data must be equal to (names, z_cls, DA_cls, err_cls, asymm_cls, ne0_cls, beta_cls, rc_out_cls, f_cls, rc_in_cls). **kwargs are the arguments of ADDMod.
    """
    
    (ma, ga, OmL, h0) = pars
    names, z_cls, DA_cls, err_cls, asymm_cls, ne0_cls, beta_cls, rc_out_cls, f_cls, rc_in_cls, Rvir_cls = data
    
    chi2 = 0.
    residuals = []
    
    for i in range(len(names)):
        
        z = z_cls[i]
        DA = DA_cls[i]
        
        ne0 = ne0_cls[i]
        rc_outer = rc_out_cls[i]
        beta_outer = beta_cls[i]
        f_inner = f_cls[i]
        rc_inner = rc_in_cls[i]
        beta_inner = beta_cls[i]
        
        if fixed_Rvir:
            r_up = 1800. # [kpc] =  1.8 Mpc for all clusters, same as Perseus
        else:
            r_up = Rvir_cls[i] # each cluster has its own virial radius, already computed under some fiducial LCDM assumption
        
        factor = ADDMod(ma, ga, z, h0, OmL,
                        ne0=ne0,
                        rc_outer=rc_outer,
                        beta_outer=beta_outer,
                        f_inner=f_inner,
                        rc_inner=rc_inner,
                        beta_inner=beta_inner,
                        r_up=r_up,
                        **kwargs)
        
        DA_th = dA_at_z(z, h0, OmL) * factor
        
        residuals.append(DA - DA_th)
    
    residuals = np.array(residuals)
    
    correction = 1.
    
    if wanna_correct:
        correction += -2.*asymm_cls * (residuals/err_cls) + 5.*asymm_cls**2. * (residuals/err_cls)**2.
    
    terms = ((residuals / err_cls)**2.)*correction
    
    chi2 = terms.sum()
    
    return chi2



##########################
# total likelihood
##########################

def lnprob(x,
           keys=None, keys_fixed=None, params=None,
           use_SH0ES=False, shoes_data=None,
           use_BOSSDR12=False, boss_data=None,
           use_BAOlowz=False, bao_data=None,
           use_Pantheon=False, pan_data=None, pan_kwargs=None,
           use_TDCOSMO=False, ext_data=None,
           use_early=False, early_data=None,
           use_clusters=False, clusters_data=None, wanna_correct=True, fixed_Rvir=False,clusters_kwargs=None,
           verbose=False):
    """
    Computes the total likelihood, as well as that for each experiment
    """
    current_point = {}
    
    for ii in range(len(keys)):
        current_point[keys[ii]] = x[ii]
    for key in keys_fixed:
        current_point[key] = params[key+' fixed']

    ma = 10**current_point['logma']
    ga = 10**current_point['logga']
    OmL = current_point['OmL']
    h0 = current_point['h0']
    
    if use_Pantheon:
        M0 = current_point['M0']
    if use_BOSSDR12:
        rs = current_point['rs']
    
    # counting the number of experiments used
    experiments_counter = sum([use_SH0ES, use_Pantheon, use_TDCOSMO, use_early, use_BOSSDR12, use_BAOlowz, use_clusters])
    lnprob_each_chi2 = []
    
    if not is_Out_of_Range(x, keys, params):  # to avoid overflow
        chi2 = 0

        # anchors
        if use_SH0ES:
            
            this_chi2 = chi2_SH0ES(M0, data=shoes_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)
            
            if verbose > 2:
                print('SHOES=%f' % this_chi2)

        # Pantheon
        if use_Pantheon:
            
            this_chi2 = chi2_Pantheon((ma, ga, OmL, h0, M0), data=pan_data, **pan_kwargs)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)
            
            if verbose > 2:
                print('pantheon=%f' % this_chi2)

        # other H0 experiments
        if use_TDCOSMO:
            
            this_chi2 = chi2_External(h0, data=ext_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)
            
            if verbose > 2:
                print('TDCOSMO=%f' % this_chi2)
            
        if use_early:
            
            this_chi2 = chi2_early(rs, data=early_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)
            
            if verbose > 2:
                print('early=%f' % this_chi2)

        # BOSS DR12
        if use_BOSSDR12:
            
            this_chi2 = chi2_BOSSDR12((OmL, h0, rs), data=boss_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)
            
            if verbose > 2:
                print('boss=%f' % this_chi2)

        # BAOlowz (6DFs + BOSS DR7 MGS, called smallz in MontePython)
        if use_BAOlowz:
            
            this_chi2 = chi2_BAOlowz((OmL, h0, rs), data=bao_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)
            
            if verbose > 2:
                print('bao=%f' % this_chi2)

        # clusters
        if use_clusters:
            
            this_chi2 = chi2_clusters((ma, ga, OmL, h0), data=clusters_data, wanna_correct=wanna_correct, fixed_Rvir=fixed_Rvir, **clusters_kwargs)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)
            
            if verbose > 2:
                print('clusters=%f' % this_chi2)

    else:
        chi2 = np.inf
        lnprob_each_chi2 = [np.inf]*experiments_counter
        
        if verbose > 2:
            print("out of range... chi2 = np.inf")

    # determine output
    res = -1./2.*chi2
    
    lnprob_each_chi2.insert(0, res)
    lnprob_each_chi2 = tuple(lnprob_each_chi2)
    
    return lnprob_each_chi2
