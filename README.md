# TumoroscopeGE
This in an extention of Tumoroscope model. It adds one more latent variable (B_kg - average expression of gene g per one cell of clone k) and one observed variable (Y_sg - gene expression of gene g in spot s).

There are 3 main classes:
* tumoroscope.py
* simulation.py
* visualization.py

and additional scripts, including:
* main_diff_config_.py and run_main_diff.sh - for running the model on simulated data
* generate_simulations.py and generate_simulations.sh - for generating the simulated data

The directory configs contains json files describing 5 different setups used to generate the simulations.
