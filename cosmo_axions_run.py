#######################################################
###       Code for emcee cosmo_axions chains        ###
###               by Chen Sun, 2020                 ###
###         and Manuel A. Buen-Abad, 2020           ###
#######################################################

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime
except:
    pass

import os
import errno
import emcee
import sys
import getopt
import warnings
import random

import numpy as np
import scipy.linalg as la

from numpy import pi, sqrt, log, log10, exp, power
from contextlib import closing

from ag_probs import omega_plasma
from igm import igm_Psurv
from icm import L_avg, icm_los_Psurv
from cosmo import H_at_z, tau_at_z, dA_at_z, muLCDM, LumMod, ADDMod

# od()
try:
    from collections import OrderedDict as od
except ImportError:
    try:
        from ordereddict import OrderedDict as od
    except ImportError:
        raise io_mp.MissingLibraryError(
            "If you are running with Python v2.5 or 2.6, you need" +
            "to manually install the ordereddict package by placing" +
            "the file ordereddict.py in your Python Path")

# numexpr, as stated in Panthon code that it's much faster than numpy
try:
    import numexpr as ne
except ImportError:
    raise io_mp.MissingLibraryError(
        "This likelihood has intensive array manipulations. You "
        "have to install the numexpr Python package. Please type:\n"
        "(sudo) pip install numexpr --user")


# CONSTANTS:
_c_ = 299792458.  # [m/s]
_alpha_ = 1./137  # fine structure constant
_me_ = 510998.95  # electron mass in eV
_1_over_cm_eV_ = 1.9732698045930252e-5  # [1/cm/eV]
_rads_over_arcsec_ = (2.*pi)/(360.*60.*60.) # [rad/arcsec]


##########################
# auxiliary functions
##########################

def read_matrix(path):
    """
    extract the matrix from the path

    This routine uses the blazing fast pandas library (0.10 seconds to load
    a 740x740 matrix). If not installed, it uses a custom routine that is
    twice as slow (but still 4 times faster than the straightforward
    numpy.loadtxt method.)

    This function is adopted from MontePython

    .. note::

        the length of the matrix is stored on the first line... then it has
        to be unwrapped. The pandas routine read_table understands this
        immediatly, though.

    """
    from pandas import read_table
    # path = os.path.join(self.data_directory, path)
    # The first line should contain the length.
    with open(path, 'r') as text:
        length = int(text.readline())

    # Note that this function does not require to skiprows, as it
    # understands the convention of writing the length in the first
    # line
    matrix = read_table(path).as_matrix().reshape((length, length))

    return matrix


def pltpath(dir, head='', ext='.pdf'):
    path = os.path.join(dir, 'plots')
    
    run_name = str(dir).rstrip('/')
    run_name = run_name.split('/')[-1]
    
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if bool(head):
        return os.path.join(path, head + '_' + run_name + ext)
    else:
        return os.path.join(path,
        'corner_' + run_name + '.pdf')


def dir_init(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return


def fill_mcmc_parameters(path):
    res = od()
    keys = []
    fixed_keys = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                pass
            elif (line.startswith('\n')) or (line.startswith('\r')):
                pass
            else:
                words = line.split("=")
                key = (words[0]).strip()
                try:
                    res[key] = float(words[1])
                except:
                    
                    # print line, words, key
                    
                    res[key] = (words[1]).strip()
                    # not a number, start parsing
                    if res[key][0] == '[' and res[key][-1] == ']':
                        # make sure the string is safe to eval()
                        res[key] = eval(res[key])
                        if res[key][3] != 0.:
                            res[key+' mean'] = res[key][0]
                            res[key+' low'] = res[key][1]
                            res[key+' up'] = res[key][2]
                            res[key+' sig'] = res[key][3]
                            keys.append(str(key))
                        else:
                            res[key+' fixed'] = res[key][0]
                            fixed_keys.append(str(key))
                    elif res[key] == 'TRUE' or res[key] == 'True' or res[key] == 'true' or res[key] == 'T' or res[key] == 'yes' or res[key] == 'Y' or res[key] == 'Yes' or res[key] == 'YES':
                        res[key] = True

                    elif res[key] == 'FALSE' or res[key] == 'False' or res[key] == 'false' or res[key] == 'F' or res[key] == 'NO' or res[key] == 'No' or res[key] == 'no' or res[key] == 'N':
                        res[key] = False
    return (res, keys, fixed_keys)


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



if __name__ == '__main__':
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    argv = sys.argv[1:]
    help_msg = 'python %s -N <number_of_steps> -o <output_folder> -L <likelihood_directory> -i <param_file> -w <number_of_walkers>' % (
        sys.argv[0])
    try:
        opts, args = getopt.getopt(argv, 'hN:o:L:i:w:')
    except getopt.GetoptError:
        raise Exception(help_msg)
    flgN = False
    flgo = False
    flgL = False
    flgi = False
    flgw = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-N':
            chainslength = arg
            flgN = True
        elif opt == '-o':
            directory = arg
            flgo = True
        elif opt == '-L':
            dir_lkl = arg
            flgL = True
        elif opt == '-i':
            path_of_param = arg
            flgi = True
        elif opt == '-w':
            number_of_walkers = int(arg)
            flgw = True
    if not (flgN and flgo and flgL and flgi and flgw):
        raise Exception(help_msg)

##########################
# initialize
##########################

    # init the dir
    dir_init(directory)

    # check if there's a preexisting param file
    if os.path.exists(os.path.join(directory, 'log.param')):
        path_of_param = os.path.join(directory, 'log.param')
        # get the mcmc params from existing file
        params, keys, keys_fixed = fill_mcmc_parameters(
            path_of_param)
    else:
        # get the mcmc params
        params, keys, keys_fixed = fill_mcmc_parameters(
            path_of_param)
        # save the input file only after the params are legit
        from shutil import copyfile
        copyfile(path_of_param, os.path.join(directory, 'log.param'))

    # fill up defaults
    try:
        params['debug']
    except KeyError:
        params['debug'] = False

    if params['debug']:
        debug = True
    else:
        debug = False

    if debug:
        print(params)
        # raise Exception('debug end')

    try:
        params['inverse_chi2']
    except KeyError:
        params['inverse_chi2'] = False
    if params['inverse_chi2'] is not True and \
       params['inverse_chi2'] is not False:
        raise Exception('Do you want to have (-1)*chi2? Please check input.param\
                        and specify the inverse_chi2 parameter with\
                        True or False')
    try:
        params['use_Pantheon']
    except KeyError:
        params['use_Pantheon'] = False
    if params['use_Pantheon'] is not True and \
       params['use_Pantheon'] is not False:
        raise Exception('Do you want Pantheon? Please check input.param\
                        and specify the use_Pantheon parameter with\
                        True or False')

    try:
        params['use_SH0ES']
    except KeyError:
        params['use_SH0ES'] = False
    if params['use_SH0ES'] is not True and \
       params['use_SH0ES'] is not False:
        raise Exception('Do you want SH0ES? Please check input.param\
                        and specify the use_SH0ES parameter with\
                        True or False')
    
    try:
        params['use_early']
    except KeyError:
        params['use_early'] = False
    if params['use_early'] is not True and \
       params['use_early'] is not False:
        raise Exception('Do you want early? Please check input.param\
                        and specify the use_early parameter with\
                        True or False')

    try:
        params['use_TDSCOSMO']
    except KeyError:
        params['use_TDSCOSMO'] = False
    if params['use_TDSCOSMO'] is not True and \
       params['use_TDSCOSMO'] is not False:
        raise Exception('Do you want TDSCOSMO? Please check input.param\
                        and specify the use_TDSCOSMO parameter with\
                        True or False')

    try:
        params['use_BOSSDR12']
    except KeyError:
        params['use_BOSSDR12'] = False
    if params['use_BOSSDR12'] is not True and \
       params['use_BOSSDR12'] is not False:
        raise Exception('Do you want BOSS DR12? Please check input.param\
                        and specify the use_BOSSDR12 parameter with\
                        True or False')

    try:
        params['use_BAOlowz']
    except KeyError:
        params['use_BAOlowz'] = False
    if params['use_BAOlowz'] is not True and \
       params['use_BAOlowz'] is not False:
        raise Exception('Do you want BOSS DR12? Please check input.param\
                        and specify the use_BAOlowz parameter with\
                        True or False')

    try:
        params['use_clusters']
    except KeyError:
        params['use_clusters'] = False
    if params['use_clusters'] is not True and \
       params['use_clusters'] is not False:
        raise Exception('Do you want clusters data? Please check input.param\
                        and specify the use_clusters parameter with\
                        True or False')
    
    try:
        wanna_correct = params['wanna_correct']
    except KeyError:
        wanna_correct = True

    try:
        redshift_dependent = params['redshift_dependent']
    except KeyError:
        redshift_dependent = True
    
    try:
        smoothed_IGM = params['smoothed_IGM']
    except KeyError:
        smoothed_IGM = False
    
    try:
        method_IGM = params['method_IGM']
    except KeyError:
        method_IGM = 'simps'
    
    try:
        Nz_IGM = params['Nz_IGM']
    except KeyError:
        Nz_IGM = 501
    
    try:
        prob_func_IGM = params['prob_func_IGM']
    except KeyError:
        prob_func_IGM = 'norm_log'
    
    try:
        omegaSN = params['omegaSN [eV]']
    except KeyError:
        omegaSN = 1.
    
    try:
        B_IGM = params['B_IGM [nG]']
    except KeyError:
        B_IGM = 1.

    try:
        ne_IGM = params['ne_IGM [1/cm3]']
    except KeyError:
        ne_IGM = 6.e-8

    try:
        s_IGM = params['s_IGM [Mpc]']
    except KeyError:
        s_IGM = 1.
    
    try:
        ICM_effect = params['ICM_effect']
    except KeyError:
        ICM_effect = False
    
    try:
        smoothed_ICM = params['smoothed_ICM']
    except KeyError:
        smoothed_ICM = True
    
    try:
        method_ICM = params['method_ICM']
    except KeyError:
        method_ICM = 'product'
    
    try:
        return_arrays = params['return_arrays']
    except KeyError:
        return_arrays = False
    
    try:
        prob_func_ICM = params['prob_func_ICM']
    except KeyError:
        prob_func_ICM = 'norm_log'
    
    try:
        Nr_ICM = params['Nr_ICM']
    except KeyError:
        Nr_ICM = 501
    
    try:
        los_method = params['los_method']
    except KeyError:
        los_method = 'quad'
    
    try:
        los_use_prepared_arrays = params['los_use_prepared_arrays']
    except KeyError:
        los_use_prepared_arrays = False
    
    try:
        los_Nr = params['los_Nr']
    except KeyError:
        los_Nr = 501

    try:
        omegaX = params['omegaX [keV]']*1.e3
    except KeyError:
        omegaX = 1.e4
    
    try:
        omegaCMB = params['omegaCMB [eV]']
    except KeyError:
        omegaCMB = 2.4e-4

    try:
        r_vir = params['R_vir [Mpc]']*1.e3
    except KeyError:
        r_vir = 1800.
    
    try:
        L_ICM = params['L_ICM [kpc]']
    except KeyError:
        L_ICM = L_avg
    
    try:
        ICM_magnetic_model = params['ICM_magnetic_model']
    except KeyError:
        ICM_magnetic_model = 'A'
    
    if ICM_magnetic_model == 'A':
        
        r_low = 10.
        B_ref = 25.
        r_ref = 0.
        eta = 0.7
    
    elif ICM_magnetic_model == 'B':
        
        r_low = 0.
        B_ref = 7.5
        r_ref = 25.
        eta = 0.5
        
    elif ICM_magnetic_model == 'C':
        
        r_low = 0.
        B_ref = 4.7
        r_ref = 0.
        eta = 0.5
        
    else:
        
        try:
            r_low = params['r_low [kpc]']
        except KeyError:
            r_low = 0.
        
        try:
            B_ref = params['B_ref [muG]']
        except KeyError:
            B_ref = 10.
        
        try:
            r_ref = params['r_ref [kpc]']
        except KeyError:
            r_ref = 0.
        
        try:
            eta = params['eta']
        except KeyError:
            eta = 0.5

##########################
# load up likelihoods
# that are read from a file
##########################

    # load SH0ES
    if params['use_SH0ES'] is True:
        # load Riess 2016
        (Anchor_SN, Anchor_SNsig, Anchor_Ceph, Anchor_Cephsig, Anchor_M,
         Anchor_Msig) = np.loadtxt(os.path.join(dir_lkl, params['anchor_lkl']),
                                   skiprows=2,
                                   delimiter=",")
        aB = params['aB']
        aBsig = params['aBsig']
        Anchor_SN = Anchor_SN - 5 * aB  # this is the measured m_SN

    # load Pantheon
    if params['use_Pantheon'] is True:
        
        PAN_lkl = np.loadtxt(os.path.join(dir_lkl, params['Pantheon_lkl']),
                             skiprows=1,
                             usecols=(1, 4, 5))
        C00 = read_matrix(os.path.join(
            dir_lkl, params['Pantheon_covmat']))
        # choose a subset of covmat and lkl
        # covmat
        full_length = len(PAN_lkl)
        subset_length = int(params['Pantheon_subset'])
        del_length = full_length - subset_length
        del_idx = np.array(random.sample(np.arange(full_length), del_length))
        C00 = np.delete(C00, del_idx, axis=1)
        C00 = np.delete(C00, del_idx, axis=0)
        # lkl
        PAN_lkl = np.delete(PAN_lkl, del_idx, axis=0)
        if params['verbose'] >= 2:
            print('full_length=%s' % full_length)
            print('subset_length=%s' % subset_length)
            print('del_length=%s' % del_length)
            print('C00.shape=%s' % str(C00.shape))
            print('PAN_lkl.shape=%s' % str(PAN_lkl.shape))
        # end of choice
        PAN_cov = ne.evaluate("C00")
        PAN_cov += np.diag(PAN_lkl[:, 2]**2)
        PAN_cov = la.cholesky(PAN_cov, lower=True, overwrite_a=True)

    # load BOSS DR12
    if params['use_BOSSDR12'] is True:
        BOSS_rsfid = params['BOSSDR12_rsfid']
        BOSS_meas_z = np.array([], 'float64')
        BOSS_meas_dM = np.array([], 'float64')
        BOSS_meas_Hz = np.array([], 'float64')
        # BOSS_meas = np.loadtxt(os.path.join(
        #     dir_lkl, params['BOSSDR12_meas']), usecols=(0, 2), skiprows=1)
        with open(os.path.join(dir_lkl, params['BOSSDR12_meas'])) as f:
            for line in f:
                words = line.split()
                if words[0] != '#':
                    if words[1] == 'dM(rsfid/rs)':
                        BOSS_meas_z = np.append(BOSS_meas_z, float(words[0]))
                        BOSS_meas_dM = np.append(BOSS_meas_dM, float(words[2]))
                    elif words[1] == 'Hz(rs/rsfid)':
                        BOSS_meas_Hz = np.append(BOSS_meas_Hz, float(words[2]))
        if debug:
            print('!!!!!!!!!!')
            print(BOSS_meas_z)
            print(BOSS_meas_dM)
            print(BOSS_meas_Hz)
        BOSS_cov = np.loadtxt(os.path.join(dir_lkl, params['BOSSDR12_covmat']))
        BOSS_icov = np.linalg.inv(BOSS_cov)

        if debug:
            print(np.shape(BOSS_icov))
            print(BOSS_icov[0])

    # load BAOlowz (6DFs + DR7 MGS)
    if params['use_BAOlowz'] is True:
        BAOlowz_meas_exp = np.array([])
        BAOlowz_meas_z = np.array([], 'float64')
        BAOlowz_meas_rs_dV = np.array([], 'float64')  # rs/dV or dV/rs
        BAOlowz_meas_sigma = np.array([], 'float64')
        BAOlowz_meas_type = np.array([], 'int')  # type 3, dV/rs, type 7 rs/dV
        with open(os.path.join(dir_lkl, params['BAOlowz_lkl'])) as f:
            for line in f:
                words = line.split()
                if line[0] != '#':
                    BAOlowz_meas_exp = np.append(BAOlowz_meas_exp, words[0])
                    BAOlowz_meas_z = np.append(BAOlowz_meas_z, float(words[1]))
                    BAOlowz_meas_rs_dV = np.append(
                        BAOlowz_meas_rs_dV, float(words[2]))
                    BAOlowz_meas_sigma = np.append(
                        BAOlowz_meas_sigma, float(words[3]))
                    BAOlowz_meas_type = np.append(
                        BAOlowz_meas_type, int(words[4]))
        if debug:
            print(BAOlowz_meas_exp)
            print(BAOlowz_meas_z)
            print(BAOlowz_meas_rs_dV)
            print(BAOlowz_meas_sigma)
            print(BAOlowz_meas_type)

# MANUEL: ADD data
    # load clusters ADD
    if params['use_clusters'] is True:

        # from Bonamente et al., astro-ph/0512349, Table 3.
        stat = np.array([0.01, 0.15, 0.08, 0.08, 0.01, 0.02])
        sys_p = np.array([0.03, 0.05, 0.075, 0.08])
        sys_n = np.array([0.05, 0.075, 0.08])

        names = []
        z_cls = np.array([])
        
        DA_cls = np.array([])
        p_err_cls = np.array([])
        n_err_cls = np.array([])
        
        ne0_cls = np.array([])
        beta_cls = np.array([])
        rc_out_cls = np.array([])
        f_cls = np.array([])
        rc_in_cls = np.array([])

        with open(dir_lkl+'add.txt', 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:

                    this_line = line.split()

                    names.append(this_line[0]+' '+this_line[1])
                    z_cls = np.append(z_cls, float(this_line[2]))
                    
                    DA_cls = np.append(DA_cls, float(this_line[3]))
                    p_err_cls = np.append(p_err_cls, float(this_line[4]))
                    n_err_cls = np.append(n_err_cls, float(this_line[5]))
                    
                    ne0_cls = np.append(ne0_cls, float(this_line[6]))
                    beta_cls = np.append(beta_cls, float(this_line[8]))
                    rc_out_cls = np.append(rc_out_cls, float(this_line[10]))
                    f_cls = np.append(f_cls, float(this_line[12]))
                    rc_in_cls = np.append(rc_in_cls, float(this_line[14]))
        
        rc_out_cls = (DA_cls*1.e3)*(_rads_over_arcsec_*rc_out_cls) # converting from arcsec to kpc
        rc_in_cls = (DA_cls*1.e3)*(_rads_over_arcsec_*rc_in_cls) # converting from arcsec to kpc

        sig_p = sqrt(DA_cls*DA_cls*((stat**2.).sum() +
                                    sys_p.sum()**2.) + p_err_cls**2.)
        sig_m = sqrt(DA_cls*DA_cls*((stat**2.).sum() +
                                    sys_n.sum()**2.) + n_err_cls**2.)

        err_cls = (sig_p + sig_m)/2.
        asymm_cls = (sig_p - sig_m)/(sig_p + sig_m)
# MANUEL: ADD data

##########################
# building likelihoods for
# each data set
##########################
    #
    # component likelihoods first
    #
    # likelihood for SH0ES anchors
    # Q: do we have a covmat for the anchors?


    def chi2_SH0ES(M0):
        chi2 = 0.
        for i in range(len(Anchor_SN)):
            chi2 += (Anchor_SN[i] - M0 - Anchor_Ceph[i])**2 / Anchor_Msig[i]**2
        return chi2

    # likelihood for BAO
    def chi2_BOSSDR12(x):
        (OmL, h0, rs) = x
        chi2 = 0.
        data_array = np.array([], 'float64')
        for i, z in enumerate(BOSS_meas_z):
            DM_at_z = tau_at_z(z, h0, OmL)  # comoving
            H_at_z_val = H_at_z(z, h0, OmL, unit='SI')  # in km/s/Mpc

            theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rs * BOSS_rsfid
            theo_H_rd_by_rdfid = H_at_z_val * rs / BOSS_rsfid

            # calculate difference between the sampled point and observations
            DM_diff = theo_DM_rdfid_by_rd_in_Mpc - \
                BOSS_meas_dM[i]
            H_diff = theo_H_rd_by_rdfid - \
                BOSS_meas_Hz[i]

            # save to data array
            data_array = np.append(data_array, DM_diff)
            data_array = np.append(data_array, H_diff)
        chi2 += np.dot(np.dot(data_array, BOSS_icov), data_array)
        # if debug:
        #     print('BOSSDR12:')
        #     print(chi2)
        return chi2

    # likelihood for BAO-lowz
    def chi2_BAOlowz(x):
        (OmL, h0, rs) = x
        chi2 = 0.
        for i, z in enumerate(BAOlowz_meas_z):
            da = dA_at_z(z, h0, OmL)
            dr = z / H_at_z(z, h0, OmL)
            dv = (da * da * (1 + z) * (1 + z) * dr)**(1. / 3.)

            if BAOlowz_meas_type[i] == 3:
                theo = dv / rs
            elif BAOlowz_meas_type[i] == 7:
                theo = rs / dv
            chi2 += ((theo - BAOlowz_meas_rs_dV[i]
                      ) / BAOlowz_meas_sigma[i]) ** 2
        if debug:
            print('BAOlowz')
            print(chi2)
        return chi2

    # likelihood for Pantheon
    def chi2_Pantheon(x):
        chi2 = 0.
        residuals = []

        # # analytically integrating out M0
        # # now one needs to incorporate coefficients of M0 and M0^2
        # # in Anchor chi2_SH0ES as well. i'm just gonna leave it for now.
        # (ma, ga, OmL, h0) = x
        # ones = np.ones(len(PAN_lkl))
        # for rec in PAN_lkl:
        #     z = rec[0]
        #     m_meas = rec[1]
        #     # sigma_meas = rec[2]
        #     residuals.append(muLCDM(z, h0, OmL) -
        #                      LumMod(ma, ga, dT(z, h0, OmL)) - m_meas)
        # L_ones = la.solve_triangular(
        #     PAN_cov, ones, lower=True, check_finite=False)
        # L_residuals = la.solve_triangular(
        #     PAN_cov, residuals, lower=True, check_finite=False)
        # A = np.dot(L_ones, L_ones)
        # B = -2.*(np.dot(L_ones, L_residuals))
        # C = np.dot(L_residuals, L_residuals)
        # chi2 = C - 1./4./A * B**2 + np.log(A)

        # numerical scan
        # analytically integrating out
        (ma, ga, OmL, h0, M0) = x
        for rec in PAN_lkl:
            z = rec[0]
            m_meas = rec[1]
            
            change = LumMod(ma, ga, z,
                            B=B_IGM,
                            mg=omega_plasma(ne_IGM),
                            h=h0,
                            OmL=OmL,
                            s=s_IGM,
                            omega=omegaSN,
                            axion_ini_frac=0.,
                            smoothed=smoothed_IGM,
                            redshift_dependent=redshift_dependent,
                            method=method_IGM,
                            prob_func=prob_func_IGM,
                            Nz=Nz_IGM)
            
            residuals.append(muLCDM(z, h0, OmL) - m_meas + M0 - change)

        L_residuals = la.solve_triangular(
            PAN_cov, residuals, lower=True, check_finite=False)
        chi2 = np.dot(L_residuals, L_residuals)
        return chi2

    # other experiments measuring H0 independently
    # TODO: add a switch later so it can be turned off more gracefully DONE
    def chi2_External(h0):
        chi2 = 0.
        # add a Gaussian prior to H0
        h0_prior_mean = params['h_TD']
        h0_prior_sig = params['h_TD_sig']
        chi2 += (h0 - h0_prior_mean)**2 / h0_prior_sig**2
        return chi2

    def chi2_early(rs):
        chi2 = 0.
        # add a Gaussian prior to rs
        rsdrag_prior_mean = params['rsdrag_mean']
        rsdrag_prior_sig = params['rsdrag_sig']
        chi2 += (rs - rsdrag_prior_mean)**2 / rsdrag_prior_sig**2
        return chi2

    def chi2_clusters(pars):

        chi2 = 0.
        residuals = []

        (ma, ga, OmL, h0) = pars

        for i in range(len(names)):

            z = z_cls[i]
            DA = DA_cls[i]
            
            ne0 = ne0_cls[i]
            rc_outer = rc_out_cls[i]
            beta_outer = beta_cls[i]
            f_inner = f_cls[i]
            rc_inner = rc_in_cls[i]
            beta_inner = beta_cls[i]

            factor = ADDMod(ma, ga, z, h0, OmL,
                            omegaX=omegaX,
                            omegaCMB=omegaCMB,
                            
                            # IGM
                            sIGM=s_IGM,
                            BIGM=B_IGM,
                            mgIGM=omega_plasma(ne_IGM),
                            smoothed_IGM=smoothed_IGM,
                            redshift_dependent=redshift_dependent,
                            method_IGM=method_IGM,
                            prob_func_IGM=prob_func_IGM,
                            Nz_IGM=Nz_IGM,
                            
                            # ICM
                            ICM_effect=ICM_effect,
                            r_low=r_low,
                            r_up=r_vir,
                            L=L_ICM,
                            smoothed_ICM=smoothed_ICM,
                            method_ICM=method_ICM,
                            return_arrays=return_arrays,
                            prob_func_ICM=prob_func_ICM,
                            Nr_ICM=Nr_ICM,
                            los_method=los_method,
                            los_use_prepared_arrays=los_use_prepared_arrays,
                            los_Nr=los_Nr,
                            B_ref=B_ref,
                            r_ref=r_ref,
                            eta=eta,
                            ne0=ne0,
                            rc_outer=rc_outer,
                            beta_outer=beta_outer,
                            f_inner=f_inner,
                            rc_inner=rc_inner,
                            beta_inner=beta_inner)

            DA_th = dA_at_z(z, h0, OmL) * factor

            residuals.append(DA - DA_th)

        residuals = np.array(residuals)

        correction = 1.

        if wanna_correct:
            correction += -2.*asymm_cls * \
                (residuals/err_cls) + 5.*asymm_cls**2. * (residuals/err_cls)**2.

        terms = ((residuals / err_cls)**2.)*correction

        chi2 = terms.sum()

        return chi2



##########################
# total likelihood
##########################

    def lnprob(x):
        current_point = {}
        for ii in range(len(keys)):
            current_point[keys[ii]] = x[ii]
        for key in keys_fixed:
            current_point[key] = params[key+' fixed']

        ma = 10**current_point['logma']
        ga = 10**current_point['logga']
        OmL = current_point['OmL']
        h0 = current_point['h0']
        if params['use_Pantheon'] is True:
            M0 = current_point['M0']
        if params['use_BOSSDR12'] is True:
            rs = current_point['rs']

        if not is_Out_of_Range(x, keys, params):  # to avoid overflow
            chi2 = 0

            # anchors
            if params['use_SH0ES'] is True:
                chi2 += chi2_SH0ES(M0)
                # print('SHOES=%f' % chi2_SH0ES(M0))

            # Pantheon
            if params['use_Pantheon'] is True:
                chi2 += chi2_Pantheon((ma, ga, OmL, h0, M0))
                # print('pantheon=%f' % chi2_Pantheon((ma, ga, OmL, h0, M0)))

            # other H0 experiments
            if params['use_TDCOSMO'] is True:
                chi2 += chi2_External(h0)
                # print('TDCOSMO=%f' % chi2_External(h0))
            
            if params['use_early'] is True:
                chi2 += chi2_early(rs)
                # print('early=%f' % chi2_early(h0))

            # BOSS DR12
            if params['use_BOSSDR12'] is True:
                chi2 += chi2_BOSSDR12((OmL, h0, rs))

            # BAOlowz (6DFs + BOSS DR7 MGS, called smallz in MontePython)
            if params['use_BAOlowz'] is True:
                chi2 += chi2_BAOlowz((OmL, h0, rs))

            # clusters
            if params['use_clusters'] is True:
                chi2 += chi2_clusters((ma, ga, OmL, h0))

        else:
            if params['inverse_chi2'] is True:
                chi2 = -np.inf
            else:
                chi2 = np.inf

        # determine output
        if params['inverse_chi2'] is True:
            res = 1./2.*chi2
        else:
            res = -1./2.*chi2
        return res

##########################
# emcee related deployment
##########################

    # initial guess
    p0mean = []
    for key in keys:
        p0mean.append(params[key+' mean'])
    if params['verbose'] > 0:
        print('keys=%s' % keys)
        print('p0mean=%s' % p0mean)
        print('keys_fixed=%s' % keys_fixed)
        for key in keys_fixed:
            print('fixed param=%s' % params[key+' fixed'])

    # initial one sigma
    p0sigma = []
    for key in keys:
        p0sigma.append(params[key+' sig'])

    ndim = len(p0mean)
    nwalkers = number_of_walkers

    # initial point, following Gaussian
    p0 = []
    for i in range(len(p0mean)):
        p0component = np.random.normal(p0mean[i], p0sigma[i], nwalkers)
        p0.append(p0component)
    p0 = np.array(p0).T

    # Set up the backend
    counter = 0
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            counter += 1
    filename = "chain_%s.h5" % (counter + 1)
    path = os.path.join(directory, filename)
    backend = emcee.backends.HDFBackend(path)
    backend.reset(nwalkers, ndim)

    flgmulti = True
    try:
        from multiprocessing import Pool
    except:
        # flgmulti = False
        # uncomment above, and comment out below to support serial running
        raise Exception('multiprocessing is not working')

    if flgmulti:
        with closing(Pool()) as pool:
            # initialize sampler
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            lnprob,
                                            backend=backend,
                                            pool=pool)
            sampler.reset()
            try:
                pos, prob, state = sampler.run_mcmc(p0,
                                                    chainslength,
                                                    progress=True)
            except Warning:
                print('p0=%s, chainslength=%s' % (p0, chainslength))
                raise
            pool.terminate()
    else:
        # initialize sampler
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        lnprob,
                                        backend=backend)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(p0, chainslength, progress=True)

    print("Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)))
