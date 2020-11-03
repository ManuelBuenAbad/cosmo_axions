#######################################################
###       Code for observational data loading       ###
###           by Manuel A. Buen-Abad, 2020          ###
###                and Chen Sun, 2020               ###
#######################################################

import os
import random
import numpy as np
import scipy.linalg as la
from numpy import pi, sqrt, log, log10, exp, power


# numexpr, as stated in Pantheon code that it's much faster than numpy
try:
    import numexpr as ne
except ImportError:
    raise io_mp.MissingLibraryError(
        "This likelihood has intensive array manipulations. You "
        "have to install the numexpr Python package. Please type:\n"
        "(sudo) pip install numexpr --user")


# CONSTANTS:
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



##########################
# data loading functions
##########################



def load_shoes(dir_lkl, anchor_lkl, aB, aBsig):
    """
    Load SH0ES.
    
    return: Anchor_SN, Anchor_SNsig, Anchor_Ceph, Anchor_Cephsig, Anchor_M, Anchor_Msig, aB, aBsig
    """
    
    (Anchor_SN, Anchor_SNsig, Anchor_Ceph,
     Anchor_Cephsig, Anchor_M, Anchor_Msig) = np.loadtxt(os.path.join(dir_lkl, anchor_lkl),
                                                         skiprows=2,
                                                         delimiter=",")
    
    Anchor_SN = Anchor_SN - 5 * aB  # this is the measured m_SN
    
    return (Anchor_SN, Anchor_SNsig, Anchor_Ceph, Anchor_Cephsig, Anchor_M, Anchor_Msig, aB, aBsig)



def load_pantheon(dir_lkl, Pantheon_lkl, Pantheon_covmat, Pantheon_subset, verbose):
    """
    Load Pantheon.
    
    return: PAN_lkl, PAN_cov
    """
    
    PAN_lkl = np.loadtxt(os.path.join(dir_lkl, Pantheon_lkl),
                         skiprows=1,
                         usecols=(1, 4, 5))
    
    C00 = read_matrix(os.path.join(dir_lkl, Pantheon_covmat))
    
    # choose a subset of covmat and lkl
    # covmat
    full_length = len(PAN_lkl)
    subset_length = int(Pantheon_subset)
    del_length = full_length - subset_length
    del_idx = np.array(random.sample(np.arange(full_length), del_length))
    C00 = np.delete(C00, del_idx, axis=1)
    C00 = np.delete(C00, del_idx, axis=0)
    
    # lkl
    PAN_lkl = np.delete(PAN_lkl, del_idx, axis=0)
    if verbose >= 2:
        print('full_length=%s' % full_length)
        print('subset_length=%s' % subset_length)
        print('del_length=%s' % del_length)
        print('C00.shape=%s' % str(C00.shape))
        print('PAN_lkl.shape=%s' % str(PAN_lkl.shape))
    # end of choice
    PAN_cov = ne.evaluate("C00")
    PAN_cov += np.diag(PAN_lkl[:, 2]**2)
    PAN_cov = la.cholesky(PAN_cov, lower=True, overwrite_a=True)
    
    return (PAN_lkl, PAN_cov)



def load_boss_dr12(dir_lkl, BOSSDR12_rsfid, BOSSDR12_meas, BOSSDR12_covmat):
    """
    Load BOSS DR12.
    
    return: BOSS_rsfid, BOSS_meas_z, BOSS_meas_dM, BOSS_meas_Hz, BOSS_cov, BOSS_icov
    """
    
    BOSS_rsfid = BOSSDR12_rsfid
    BOSS_meas_z = np.array([], 'float64')
    BOSS_meas_dM = np.array([], 'float64')
    BOSS_meas_Hz = np.array([], 'float64')
    
    
    with open(os.path.join(dir_lkl, BOSSDR12_meas)) as f:
        for line in f:
            words = line.split()
            if words[0] != '#':
                if words[1] == 'dM(rsfid/rs)':
                    BOSS_meas_z = np.append(BOSS_meas_z, float(words[0]))
                    BOSS_meas_dM = np.append(BOSS_meas_dM, float(words[2]))
                elif words[1] == 'Hz(rs/rsfid)':
                    BOSS_meas_Hz = np.append(BOSS_meas_Hz, float(words[2]))
    
    BOSS_cov = np.loadtxt(os.path.join(dir_lkl, BOSSDR12_covmat))
    BOSS_icov = np.linalg.inv(BOSS_cov)

    return (BOSS_rsfid, BOSS_meas_z, BOSS_meas_dM, BOSS_meas_Hz, BOSS_cov, BOSS_icov)




def load_bao_lowz(dir_lkl, BAOlowz_lkl):
    """
    Load BAOlowz (6DFs + DR7 MGS)
    
    return: BAOlowz_meas_exp, BAOlowz_meas_z, BAOlowz_meas_rs_dV, BAOlowz_meas_sigma, BAOlowz_meas_type
    """
    
    BAOlowz_meas_exp = np.array([])
    BAOlowz_meas_z = np.array([], 'float64')
    BAOlowz_meas_rs_dV = np.array([], 'float64')  # rs/dV or dV/rs
    BAOlowz_meas_sigma = np.array([], 'float64')
    BAOlowz_meas_type = np.array([], 'int')  # type 3, dV/rs, type 7 rs/dV
    
    with open(os.path.join(dir_lkl, BAOlowz_lkl)) as f:
        
        for line in f:
            words = line.split()
            
            if line[0] != '#':
                BAOlowz_meas_exp = np.append(BAOlowz_meas_exp, words[0])
                BAOlowz_meas_z = np.append(BAOlowz_meas_z, float(words[1]))
                BAOlowz_meas_rs_dV = np.append(BAOlowz_meas_rs_dV, float(words[2]))
                BAOlowz_meas_sigma = np.append(BAOlowz_meas_sigma, float(words[3]))
                BAOlowz_meas_type = np.append(BAOlowz_meas_type, int(words[4]))
    
    return (BAOlowz_meas_exp, BAOlowz_meas_z, BAOlowz_meas_rs_dV, BAOlowz_meas_sigma, BAOlowz_meas_type)


        
def load_clusters(dir_lkl):
    """
    Load clusters ADD.
    
    return: names, z_cls, DA_cls, err_cls, asymm_cls, ne0_cls, beta_cls, rc_out_cls, f_cls, rc_in_cls
    """
    
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
    
    sig_p = sqrt(DA_cls*DA_cls*((stat**2.).sum() + sys_p.sum()**2.) + p_err_cls**2.)
    sig_m = sqrt(DA_cls*DA_cls*((stat**2.).sum() + sys_n.sum()**2.) + n_err_cls**2.)
    
    err_cls = (sig_p + sig_m)/2.
    asymm_cls = (sig_p - sig_m)/(sig_p + sig_m)
    
    return (names, z_cls, DA_cls, err_cls, asymm_cls, ne0_cls, beta_cls, rc_out_cls, f_cls, rc_in_cls)
