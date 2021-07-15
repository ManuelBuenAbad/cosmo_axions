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
from icm import L_avg

import data
import chi2

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


##########################
# auxiliary functions
##########################


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
        params['use_TDCOSMO']
    except KeyError:
        params['use_TDCOSMO'] = False
    if params['use_TDCOSMO'] is not True and \
       params['use_TDCOSMO'] is not False:
        raise Exception('Do you want TDCOSMO? Please check input.param\
                        and specify the use_TDCOSMO parameter with\
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
        fixed_Rvir = params['fixed_Rvir']
    except KeyError:
        fixed_Rvir = False

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

    try:
        mu = params['signal_strength']
    except KeyError:
        mu = 1.

    # consolidating the keyword parameters for some likelihoods
    pan_kwargs = {'B':B_IGM,
                  'mg':omega_plasma(ne_IGM),
                  's':s_IGM,
                  'omega':omegaSN,
                  'axion_ini_frac':0.,
                  'smoothed':smoothed_IGM,
                  'redshift_dependent':redshift_dependent,
                  'method':method_IGM,
                  'prob_func':prob_func_IGM,
                  'Nz':Nz_IGM,
                  'mu':mu}

    clusters_kwargs = {'omegaX':omegaX,
                      'omegaCMB':omegaCMB,
                      # IGM
                      'sIGM':s_IGM,
                      'BIGM':B_IGM,
                      'mgIGM':omega_plasma(ne_IGM),
                      'smoothed_IGM':smoothed_IGM,
                      'redshift_dependent':redshift_dependent,
                      'method_IGM':method_IGM,
                      'prob_func_IGM':prob_func_IGM,
                      'Nz_IGM':Nz_IGM,
                      # ICM
                      'ICM_effect':ICM_effect,
                      'r_low':r_low,
                      'L':L_ICM,
                      'smoothed_ICM':smoothed_ICM,
                      'method_ICM':method_ICM,
                      'return_arrays':return_arrays,
                      'prob_func_ICM':prob_func_ICM,
                      'Nr_ICM':Nr_ICM,
                      'los_method':los_method,
                      'los_use_prepared_arrays':los_use_prepared_arrays,
                      'los_Nr':los_Nr,
                      'mu':mu,
                      'B_ref':B_ref,
                      'r_ref':r_ref,
                      'eta':eta}

##########################
# load up likelihoods
# that are read from a file
##########################

    experiments = [] # a list of shorthand names for the experiments

    # load SH0ES
    if params['use_SH0ES'] is True:
        shoes_data = data.load_shoes(dir_lkl,
                                params['anchor_lkl'],
                                params['aB'],
                                params['aBsig'])
        experiments.append('shoes')
    else:
        shoes_data = None


    # load Pantheon
    if params['use_Pantheon'] is True:
        pan_data = data.load_pantheon(dir_lkl,
                                 params['Pantheon_lkl'],
                                 params['Pantheon_covmat'],
                                 params['Pantheon_subset'],
                                 params['verbose'])
        experiments.append('pantheon')
    else:
        pan_data = None


    # load BOSS DR12
    if params['use_BOSSDR12'] is True:
        boss_data = data.load_boss_dr12(dir_lkl,
                                   params['BOSSDR12_rsfid'],
                                   params['BOSSDR12_meas'],
                                   params['BOSSDR12_covmat'])
        experiments.append('boss')
    else:
        boss_data = None


    # load BAOlowz (6DFs + DR7 MGS)
    if params['use_BAOlowz'] is True:
        bao_data = data.load_bao_lowz(dir_lkl,
                                 params['BAOlowz_lkl'])
        experiments.append('bao')
    else:
        bao_data = None

    # load H0 data
    if params['use_TDCOSMO'] is True:
        ext_data = (params['h_TD'], params['h_TD_sig'])
        experiments.append('tdcosmo')
    else:
        ext_data = None

    # load rsdrag data
    if params['use_early'] is True:
        early_data = (params['rsdrag_mean'], params['rsdrag_sig'])
        experiments.append('planck')
    else:
        early_data = None

    # load clusters ADD
    if params['use_clusters'] is True:
        clusters_data = data.load_clusters(dir_lkl)
        experiments.append('clusters')
    else:
        clusters_data = None


##########################
# emcee related deployment
##########################

    def lnprob(x):
        """
        Defining lnprob at the top level, to avoid Pickle errors.
        """

        return chi2.lnprob(x,
                           keys=keys, keys_fixed=keys_fixed, params=params,
                           use_SH0ES=params['use_SH0ES'], shoes_data=shoes_data,
                           use_BOSSDR12=params['use_BOSSDR12'], boss_data=boss_data,
                           use_BAOlowz=params['use_BAOlowz'], bao_data=bao_data,
                           use_Pantheon=params['use_Pantheon'], pan_data=pan_data, pan_kwargs=pan_kwargs,
                           use_TDCOSMO=params['use_TDCOSMO'], ext_data=ext_data,
                           use_early=params['use_early'], early_data=early_data,
                           use_clusters=params['use_clusters'], clusters_data=clusters_data, wanna_correct=wanna_correct, fixed_Rvir=fixed_Rvir, clusters_kwargs=clusters_kwargs,
                           verbose=params['verbose'])

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

    # the names and types of the blobs
    dtype = [(exper, float) for exper in experiments]

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
                                            pool=pool,
                                            blobs_dtype=dtype)
            sampler.reset()

            try:
                result = sampler.run_mcmc(p0, chainslength, progress=True)

            except Warning:
                print('p0=%s, chainslength=%s' % (p0, chainslength))
                raise

            pool.terminate()
    else:
        # initialize sampler
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        lnprob,
                                        backend=backend,
                                        blobs_dtype=dtype)
        sampler.reset()

        result = sampler.run_mcmc(p0, chainslength, progress=True)

    print("Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)))
