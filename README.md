# cosmo_axions
A Bayesian Python code to fit the axion-photon parameter space to cosmological data.

Written by Manuel A. Buen-Abad and Chen Sun, 2020

Requirements
-----------------------------------------

1. Python  
2. numpy  
3. scipy  
4. emcee  
5. corner  


How to run
-----------------------------------------

In the terminal:

	$ python cosmo_axions_run.py -L likelihoods/ -o path/to/your/chain/output/ -i inputs/the_param_file.param -N number_of_points -w number_of_walkers

After the runs are finished, you can analyze them with:

	$ python cosmo_axions_analysis.py -i path/to/your/chain/output/

Once the analysis is done, if you wanna output the contours in ma-ga space from the frequentist likelihood ratio test, do:

	$ python bin_chi2.py -c path/to/your/chain/output/ -b number_of_ma-ga_bins

where the argument with flag -b bins the ma-ga parameter space in order to minimize the chi2 in each bin. A value of ~50 is good enough.


Bibtex entry
-----------------------------------------

If you use this code or find it in any way useful for your research, please cite [Buen-Abad, Fan, & Sun (2020)](https://arxiv.org/abs/2011.05993). The Bibtex entry is:

	@article{Buen-Abad:2020zbd,
	    author = "Buen-Abad, Manuel A. and Fan, JiJi and Sun, Chen",
	    title = "{Constraints on Axions from Cosmic Distance Measurements}",
	    eprint = "2011.05993",
	    archivePrefix = "arXiv",
	    primaryClass = "hep-ph",
	    month = "11",
	    year = "2020"
	}
