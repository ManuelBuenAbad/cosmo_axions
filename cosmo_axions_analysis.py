#######################################################
###      Code for emcee cosmo_axions analysis       ###
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
import numpy as np
import emcee
from emcee.autocorr import AutocorrError
import corner
import h5py
import sys
import getopt
from cosmo_axions_run import pltpath, fill_mcmc_parameters

if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, 'hi:')
    except getopt.GetoptError:
        raise Exception('python %s -i <folder_of_chains>' %
                        (sys.argv[0]))
    flgi = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception('python %s -i <folder_of_chains>' %
                            (sys.argv[0]))
        elif opt == '-i':
            directory = arg
            flgi = True
    if not flgi:
        raise Exception('python %s -i <folder_of_chains>' %
                        (sys.argv[0]))

    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            path = os.path.join(directory, filename)

            reader = emcee.backends.HDFBackend(path, read_only=True)
            # tau = reader.get_autocorr_time()
            try:
                tau = reader.get_autocorr_time()
                print('auto correlation time = %s' % tau)
            except AutocorrError as e:
                # this is the case the chain is shorter than 50*(autocorr time)
                print('%s' % e)
                # tau = [410., 100., 140, 140]
                tau = e.tau
                print('setting correlation time to the current estimate.')

            # use auto-correlation time to estimate burnin here
            # works only for long chains
            burnin = int(2*np.max(tau))
            thin = int(0.5*np.min(tau))
            samples = reader.get_chain(
                discard=burnin, flat=True, thin=thin)
            print("burn-in: {0}".format(burnin))
            print("thin: {0}".format(thin))
            print("flat chain shape: {0}".format(samples.shape))
            try:
                all_samples = np.append(all_samples, samples, axis=0)
            except:
                all_samples = samples
        else:
            continue

    # load log.param
    params, keys, keys_fixed = fill_mcmc_parameters(
        os.path.join(directory, 'log.param'))

    # test data authenticity
    if len(keys) != len(samples[0]):
        raise Exception(
            'log.param and h5 files are not consistent. Data is compromised. Quit analyzing.')

    # compute mean
    dim_of_param = len(samples[0])
    mean = np.mean(samples, axis=0)
    print('mean = %s' % mean)

    # corner plot
    plt.figure(0)
    # labels = keys
    labels = [r"$\Omega_\Lambda$", r"$h$", r"$\log\ m_a$", r"$\log\ g_a$"]
    if 'M0' in keys:
        labels.append(r"$M_0$")
    if 'rs' in keys:
        labels.append(r"$r_s^{drag}$")

    figure = corner.corner(samples, labels=labels, quantiles=[
                           0.16, 0.5, 0.84], show_titles=True,
                           title_kwargs={"fontsize": 12})
    axes = np.array(figure.axes).reshape((dim_of_param, dim_of_param))
    
    plt.savefig(pltpath(directory))

    # focusing on ma-ga
    plt.figure(1)
    reduced_labels = [r"$\log\ m_a$", r"$\log\ g_a$"]
    reduced_samples = samples[:,2:4]
    reduced_dim = len(reduced_labels)

    figure = corner.corner(reduced_samples, labels=reduced_labels,
                           quantiles=[0.16, 0.5, 0.84],
                           color='r', show_titles=True,
                           plot_datapoints=False,
                           plot_density=False,
                           levels=[1.-np.exp(-(2.)**2 /2.)],
                           title_kwargs={"fontsize": 12},
                           hist_kwargs={'color':None})
    axes = np.array(figure.axes).reshape((reduced_dim, reduced_dim))

    p = (figure.axes)[2].collections[0].get_paths()[0]
    v = p.vertices
    
    np.savetxt(pltpath(directory, head='all_pts', ext='.txt'), v)
    
    # cutting out the unnecessary points
    v = v[~(v[:,0] < -17.), :]
    v = v[~(v[:,1] < -13.), :]
    v = v[~(v[:,0] > -11.), :]

    np.savetxt(pltpath(directory, head='reduced_pts', ext='.txt'), v)

    plt.savefig(pltpath(directory, head='custom'))
