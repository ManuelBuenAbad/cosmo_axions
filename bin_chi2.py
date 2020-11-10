#######################################################
###          Code for binned chi2(ma, ga)           ###
###          by Manuel A. Buen-Abad, 2020           ###
###               and Chen Sun, 2020                ###
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
import sys
import getopt
import warnings
import random
import h5py

import numpy as np
from numpy import pi, sqrt, log, log10, exp, power
from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator as lndi
from tqdm import tqdm
from cosmo_axions_run import pltpath


##########################
# auxiliary functions
##########################

if __name__ == '__main__':
    
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    argv = sys.argv[1:]
    help_msg = 'python %s -c <chain_folder> -b <bins>' % (
        sys.argv[0])
    try:
        opts, args = getopt.getopt(argv, 'h:c:b:')
    except getopt.GetoptError:
        raise Exception(help_msg)
    flgc = False
    flgb = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-c':
            directory = arg
            flgc = True
        elif opt == '-b':
            bins = int(arg)
            flgb = True
    if not (flgc and flgb):
        raise Exception(help_msg)
    
    # reading chains
    path = directory+'chain_1.h5'
    f = h5py.File(path, 'r')
    
    f = f['mcmc']
    keys = f.keys()
    print keys
    
    #idx = np.random.randint(100)
    #print f['chain'][idx,0], -2*f['log_prob'][idx,0], f['blobs']['pantheon'][idx,0]
    
    pts = np.array(f['chain']) # the points
    pts = pts.reshape(-1, 6)
    
    chi2_tot = np.array(f['log_prob'])
    chi2_tot *= -2
    chi2_tot = chi2_tot.reshape(-1)
    
    blobs = f['blobs']
    experiments = dict(blobs.dtype.fields).keys()
    
    each_chi2 = {exper:blobs[exper].reshape(-1) for exper in experiments} # the experiments' chi2s for each point
    
    del f
    
    bf_chi2, bf_idx = min(chi2_tot), chi2_tot.argmin() # the best fit chi2 and where it is
    each_sum = sum([each_chi2[exper][bf_idx] for exper in experiments]) # the sum of the chi2 from each experiment at the best fit point
    
    print "chi2 best fit: {} = {}".format(bf_chi2, each_sum) # sanity check
    
    # binning the (ma, ga) parameter space:
    
    chain_ga = pts[:,3] # the values of ga
    chain_neg_ga = chain_ga[np.where(chain_ga<0)] # only negatives!
    _, edges_ga = np.histogram(chain_neg_ga, bins=bins) # the edges of the bins
    
    chain_ma = pts[:,2] # the values of ma
    chain_neg_ma = chain_ma[np.where(chain_ma<0)] # only negatives!
    _, edges_ma = np.histogram(chain_neg_ma, bins=bins) # the edges of the bins
    
    # the center values
    block_ga = (edges_ga[:-1] + edges_ga[1:])/2.
    block_ma = (edges_ma[:-1] + edges_ma[1:])/2.
    mesh_ga, mesh_ma = np.meshgrid(block_ga, block_ma, indexing='ij')
    
    # preparation for the computation of the chi2(ma, ga) function
    
    chi2_mins = [] # the min chi2
    idx_mins = [] # the index of the min chi2
    ma_ga_chi2 = [] # the triples (ma, ga, min_chi2) only for those bins where the value is well defined
    
    wheres = {} # those indices that satisfy the conditions to be within the bin
    
    for i in tqdm(range(len(edges_ga)-1)):
        for j in range(len(edges_ma)-1):
            
            # those points with ga, ma values within the bin
            wheres[i,j] = np.where((chain_ga>edges_ga[i])
                                   & (chain_ga<edges_ga[i+1])
                                   & (chain_ma>edges_ma[j])
                                   & (chain_ma<edges_ma[j+1]))
            
            #print wheres[i,j]
            
            # the chi2s in that bin
            chi2_block =  chi2_tot[wheres[i,j]]
            
            # appending minima and indices
            if len(chi2_block)>0:
                
                this_min_chi2 = min(chi2_block) # the minimum chi2 of this bin
                
                # appending to the list
                chi2_mins.append(this_min_chi2)
                idx_mins.append(chi2_block.argmin())
                # appending to the data
                ma_ga_chi2.append([mesh_ma[i,j], mesh_ga[i,j], this_min_chi2])
            
            else:
                chi2_mins.append(np.inf)
                idx_mins.append(-1)
                
                continue
    
    # converting to numpy arrays
    chi2_mins = np.array(chi2_mins)
    idx_mins = np.array(idx_mins, dtype=int)
    
    chi2_mins = chi2_mins.reshape(mesh_ma.shape)
    idx_mins = idx_mins.reshape(mesh_ma.shape)
    
    ma_ga_chi2 = np.array(ma_ga_chi2)
    
    # interpolating over the data
    delta_chi2 = lndi(ma_ga_chi2[:,0:2], ma_ga_chi2[:,2]-bf_chi2) # since data is not a uniform grid, we need to use LinearNDInterpolator
    
    # plot
    plt.figure(101)
    plt.xlabel(r'$\log_{10} m_a$')
    plt.ylabel(r'$\log_{10} g_a$')
    plt.xlim(-17., -11.)
    plt.ylim(-13., -8.)
    plt.title(r'$\Delta \chi^2$ contours')
    
    ma_arr = np.linspace(edges_ma[0], edges_ma[-1], 101)
    ga_arr = np.linspace(edges_ga[0], edges_ga[-1], 101)
    ga_gr, ma_gr = np.meshgrid(ga_arr, ma_arr, indexing='ij')
    delta_arr = delta_chi2(ma_gr, ga_gr)
    
    # the delta_chi2 1- and 2-sigma contours, both straight out of the data and interpolated
    plt.contour(mesh_ma, mesh_ga, chi2_mins-bf_chi2, levels=[2.29141, 6.15823], colors=['b', 'r'])
    plt.contour(ma_arr, ga_arr, delta_arr, levels=[2.29141, 6.15823], colors=['cyan', 'orange'], linestyles=[':', ':'])
    plt.savefig(pltpath(directory, head='delta_chi2_contours'))
    
    # the points of the 2-sigma (95.45% C.L.) contour
    cs = plt.contour(ma_arr, ga_arr, delta_arr, levels=[6.15823])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    np.savetxt(pltpath(directory, head='2sigma_pts', ext='.txt'), v)
    
    # the points of the 95% C.L. contour
    cs2 = plt.contour(ma_arr, ga_arr, delta_arr, levels=[5.99146])
    p2 = cs2.collections[0].get_paths()[0]
    v2 = p2.vertices
    np.savetxt(pltpath(directory, head='95CL_pts', ext='.txt'), v2)
