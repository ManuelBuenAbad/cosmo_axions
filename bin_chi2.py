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
from scipy.interpolate import interp1d, interp2d
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
    help_msg = 'python %s -c <chain_folder> -b <bins> -n <negative_signal_chain_folder>' % (
        sys.argv[0])
    try:
        opts, args = getopt.getopt(argv, 'h:c:b:n:')
    except getopt.GetoptError:
        raise Exception(help_msg)
    flgc = False
    flgb = False
    flgn = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-c':
            directory = arg
            flgc = True
        elif opt == '-b':
            bins = int(arg)
            flgb = True
        elif opt == '-n':
            neg_dir = arg
            flgn = True
    if not (flgc and flgb):
        raise Exception(help_msg)

    # reading chains
    path = directory+'chain_1.h5'
    f = h5py.File(path, 'r')

    f = f['mcmc']
    keys = f.keys()
    print keys

    pts = np.array(f['chain']) # the points
    pts = pts.reshape(-1, 6)
    
#     print pts.shape

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
    
    # if we passed a chain with negative signal strength, we need to read that data
    
    if flgn:
        
        print "Preparing to read the chain with negative strength signal, located in {}".format(neg_dir)
        
        neg_path = neg_dir+'chain_1.h5'
        nf = h5py.File(neg_path, 'r')
        
        nf = nf['mcmc']
        nkeys = nf.keys()
        
        npts = np.array(nf['chain'])
        npts = npts.reshape(-1, 6)
        
#         print npts.shape
        
        nchi2_tot = np.array(nf['log_prob'])
        nchi2_tot *= -2
        nchi2_tot = nchi2_tot.reshape(-1)
        
        nblobs = nf['blobs']
        nexperiments = dict(nblobs.dtype.fields).keys()
        
        neach_chi2 = {nexper:nblobs[nexper].reshape(-1) for nexper in nexperiments} # the experiments' chi2s for each point
        
        del nf
        
        nbf_chi2, nbf_idx = min(nchi2_tot), nchi2_tot.argmin() # the best fit chi2 and where it is
        neach_sum = sum([neach_chi2[nexper][nbf_idx] for nexper in nexperiments]) # the sum of the chi2 from each experiment at the best fit point
        
        print "chi2 best fit for negative signal strength: {} = {}".format(nbf_chi2, neach_sum) # sanity check
    
    #---------------------------------
    # PART 1: ga & ma: points and bins
    #..................... ...........
    
    # ga:
    chain_ga = pts[:,3] # the values of ga
    chain_neg_ga = chain_ga[np.where(chain_ga<0)] # only negatives!
    _, edges_ga = np.histogram(chain_neg_ga, bins=bins) # the edges of the bins
    block_ga = (edges_ga[:-1] + edges_ga[1:])/2. # center values
    
    # ma:
    chain_ma = pts[:,2] # the values of ma
    chain_neg_ma = chain_ma[np.where(chain_ma<0)] # only negatives!
    _, edges_ma = np.histogram(chain_neg_ma, bins=bins) # the edges of the bins
    block_ma = (edges_ma[:-1] + edges_ma[1:])/2. # center values
    
    # ma-ga:
    # mesh of (ma, ga) parameter space blocks
    mesh_ga, mesh_ma = np.meshgrid(block_ga, block_ma, indexing='ij')
    
#     print edges_ma.shape, block_ma.shape
    
    
    #---------------------
    # PART 2: chi2(ma, ga)
    #.....................
    
    # 2.a block analysis of (ma, ga) 2D space and chi2

    # preparing the arrays over the 2D (ma, ga) parameter space
    chi2_mins_2D = [] # the min chi2 in the 2D space
    idx_mins_2D = [] # the index of the min chi2 in the 2D space
    ma_ga_chi2 = [] # the triples (ma, ga, min_chi2) only for those bins where the value is well defined

    wheres_2D = {} # those location indices whose (ma, ga) parameter values are within the bin
    
    print "Computing 2D chi2(ma, ga)"
    
    for i in tqdm(range(len(edges_ga)-1)):
        for j in range(len(edges_ma)-1):

            # those points with ga, ma values within the bin (i, j)
            wheres_2D[i,j] = np.where((chain_ga>edges_ga[i])
                                      & (chain_ga<edges_ga[i+1])
                                      & (chain_ma>edges_ma[j])
                                      & (chain_ma<edges_ma[j+1]))

            # the chi2s in the ij-th bin
            chi2_ij_block =  chi2_tot[wheres_2D[i,j]]

            # appending minima and indices
            if len(chi2_ij_block) > 0:

                min_chi2_ij = min(chi2_ij_block) # the minimum chi2 of this bin

                # appending to the list
                chi2_mins_2D.append(min_chi2_ij)
                idx_mins_2D.append(chi2_ij_block.argmin())
                # appending to the data
                ma_ga_chi2.append([mesh_ma[i,j], mesh_ga[i,j], min_chi2_ij])

            else:
                chi2_mins_2D.append(np.inf)
                idx_mins_2D.append(-1)

                continue

    
    # 2.b finding chi2(ma, ga)
    
    # converting to numpy arrays
    chi2_mins_2D = np.array(chi2_mins_2D)
    idx_mins_2D = np.array(idx_mins_2D, dtype=int)

    chi2_mins_2D = chi2_mins_2D.reshape(mesh_ma.shape)
    idx_mins_2D = idx_mins_2D.reshape(mesh_ma.shape)

    ma_ga_chi2 = np.array(ma_ga_chi2)
    
#     print ma_ga_chi2.shape
#     print ma_ga_chi2[:,0:2].shape
#     print ma_ga_chi2[:,2].shape
    # interpolating over the data
    chi2_ma_ga_fn = lndi(ma_ga_chi2[:,0:2], ma_ga_chi2[:,2]) # since data is not a uniform grid, we need to use LinearNDInterpolator
    
    
    #---------------------
    # PART 3: chi2(ma)
    #.....................
    
    # 3.a block analysis of ma-space and chi2
    
    # preparing the arrays over the 1D ma-parameter space
    chi2_mins_1D = [] # the min chi2 in the 1D space
    idx_mins_1D = [] # the index of the min chi2 in the 1D space
    ma_chi2 = [] # the doubles (ma, min_chi2) only for those bins where the value is well defined
    
    print "Computing 1D chi2(ma)"
    
    for i in tqdm(range(len(edges_ma)-1)):
        
        # locations in the chain whose ma's are within the i-th ma-bin
        where = np.where((chain_ma>edges_ma[i])
                         & (chain_ma<edges_ma[i+1]))
        
        # the chi2s in that bin
        chi2_i_block =  chi2_tot[where]
        
#         print where
        
        # appending minima and indices
        if len(chi2_i_block)>0:
            
            min_chi2_i = min(chi2_i_block) # the minimum chi2 of this bin
            
            # appending to the list
            chi2_mins_1D.append(min_chi2_i)
            idx_mins_1D.append(chi2_i_block.argmin())
            # appending to the data
            ma_chi2.append([block_ma[i], min_chi2_i])
        
        else:
            chi2_mins_1D.append(np.inf)
            idx_mins_1D.append(-1)
            
            continue
    
    # 3.b finding chi2(ma)
    
    # converting to numpy arrays
    chi2_mins_1D = np.array(chi2_mins_1D)
    idx_mins_1D = np.array(idx_mins_1D, dtype=int)

    chi2_mins_1D = chi2_mins_1D.reshape(block_ma.shape)
    idx_mins_1D = idx_mins_1D.reshape(block_ma.shape)

    ma_chi2 = np.array(ma_chi2)

    # interpolating over the data
#     print ma_chi2.shape
    chi2_ma_fn = interp1d(ma_chi2[:,0], ma_chi2[:,-1], fill_value="extrapolate")
    
    
    # if we passed a shain with negative signal strength, we need to include it in our statistical analysis
    if flgn:
        
        #----------------------------------------------
        # PART 4: chi2(ma) for negative signal strength
        #..............................................
        
        # 4.a block analysis of ma-space and nchi2. NOTE: it has to be the same mass bins!!!
        
        nchain_ma = npts[:,2] # the values of ma
        
        # preparing the arrays over the 1D ma-parameter space
        nchi2_mins_1D = [] # the min chi2 in the 1D space
        nidx_mins_1D = [] # the index of the min chi2 in the 1D space
        nma_chi2 = [] # the doubles (ma, min_chi2) only for those bins where the value is well defined
        
        
        print "Computing 1D chi2(ma), for negative signal strength"
        
        for i in tqdm(range(len(edges_ma)-1)): # NOTE: it has to be the same mass bins!!!
            
            # locations in the chain whose ma's are within the i-th ma-bin
            where = np.where((nchain_ma>edges_ma[i])
                             & (nchain_ma<edges_ma[i+1]))
            
            # the chi2s in that bin
            nchi2_i_block =  nchi2_tot[where]
            
            # appending minima and indices
            if len(nchi2_i_block)>0:
                
                nmin_chi2_i = min(nchi2_i_block) # the minimum chi2 of this bin
                
                # appending to the list
                nchi2_mins_1D.append(nmin_chi2_i)
                nidx_mins_1D.append(nchi2_i_block.argmin())
                # appending to the data
                nma_chi2.append([block_ma[i], nmin_chi2_i])
            
            else:
                nchi2_mins_1D.append(np.inf)
                nidx_mins_1D.append(-1)
                
                continue
        
        # 4.b finding nchi2(ma)
        
        # converting to numpy arrays
        nchi2_mins_1D = np.array(nchi2_mins_1D)
        nidx_mins_1D = np.array(nidx_mins_1D, dtype=int)
        
        nchi2_mins_1D = nchi2_mins_1D.reshape(block_ma.shape)
        nidx_mins_1D = nidx_mins_1D.reshape(block_ma.shape)
        
        nma_chi2 = np.array(nma_chi2)
        
        # interpolating over the data
        nchi2_ma_fn = interp1d(nma_chi2[:,0], nma_chi2[:,-1], fill_value="extrapolate")
    
    
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
    
    # defining the fixed-ma best-fit chi2
    if not flgn:
        bf_chi2_ma = chi2_ma_fn(ma_gr)
        neg_str = ""
    else:
        bf_chi2_ma = np.minimum.reduce([chi2_ma_fn(ma_gr), nchi2_ma_fn(ma_gr)])
        neg_str = "_neg-sig"
    
    # array of delta chi2
    delta_arr = chi2_ma_ga_fn(ma_gr, ga_gr) - bf_chi2_ma
    
    # the points of the 2-sigma (95% C.L.) contour for a one-sided test (2.705543 chi2 threshold)
    cs = plt.contour(ma_arr, ga_arr, delta_arr, levels=[2.705543])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    np.savetxt(pltpath(directory, head='one-sided_95CL_pts'+neg_str, ext='.txt'), v)
    
    # the delta_chi2 contour
    plt.contour(ma_arr, ga_arr, delta_arr, levels=[2.705543], colors=['blue'], linestyles=['-'])
    
    if flgn: # make the same plot but without the chi2_min from the negative signal strength
        plt.contour(ma_arr, ga_arr, (chi2_ma_ga_fn(ma_gr, ga_gr) - chi2_ma_fn(ma_gr)), levels=[2.705543], colors=['red'], linestyles=['--'])
    
    plt.savefig(pltpath(directory, head='delta_chi2_contour'+neg_str))