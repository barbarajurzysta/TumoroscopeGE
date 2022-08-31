# TumoroscopeGE
This in an extention of Tumoroscope model. It adds one more latent variable (B_kg - average expression of gene g per one cell of clone k) and one observed variable (Y_sg - gene expression of gene g in spot s).

There are 3 main classes:
* tumoroscope.py
* simulation.py
* visualization.py

and additional scripts, including:
* main_diff_config_.py and run_main_diff.sh - for running TumoroscopeGE on simulated data
* generate_simulations.py and generate_simulations.sh - for generating the simulated data
* run_tumoroscope1000.py and run_tum.sh - for running TumoroscopeGE on real data

If the name of the file includes "_basic" or "_only_tum", that means it is used to apply the basic Tumoroscope.  
If the name of the file includes "combined", that means it is used to apply the combined model (first running Tumoroscope and then TumoroscopeGE).

The directory configs contains json files describing 6 different setups used to generate the simulations.  
The directory prostate_tumoroscope_input contains files necessary for running models on the real data (prostate cancer dataset [1])

[1] Berglund, Emelie, et al. "Spatial maps of prostate cancer transcriptomes reveal an unexplored landscape of heterogeneity." *Nature communications* 9.1 (2018): 1-13.
