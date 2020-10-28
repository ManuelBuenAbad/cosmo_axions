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


Bibtex entry
-----------------------------------------

If you use this code, cite the following paper:

@article{}
