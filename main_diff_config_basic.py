import tumoroscope_basic as tum
import constants
import pickle
import numpy as np
import random
from multiprocessing import Pool
import time
import json
import re
import pandas as pd
import glob, os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def run_in_parallel(args):
    return (args.gibbs_sampling(seed=random.randint(1, 1000), min_iter=min_iter, max_iter=max_iter,burn_in=burn_in,
                                batch=batch, simulated_data=sample_1, n_sampling=True, F_fraction=F_fraction, 
                                theta_variable=theta_variable, pi_2D=pi_2D, th=th, every_n_sample=every_n_sample, 
                                changes_batch=changes_batch, var_calculation=var_calculation))

def run_in_parallel_n(args):
    return (args.gibbs_sampling(seed=random.randint(1, 1000), min_iter=min_iter, max_iter=max_iter,burn_in=burn_in,
                                batch=batch, simulated_data=sample_1, n_sampling=False, F_fraction=F_fraction,
                                theta_variable=theta_variable, pi_2D=pi_2D, th=th, every_n_sample=every_n_sample, 
                                changes_batch=changes_batch, var_calculation=var_calculation))

pi_2D = True
th = 0.8  # threshhold for Z
constants.CHAINS = 5
constants.CORES = 35
every_n_sample = 5
changes_batch = 500


onLaptop = 'False'
if onLaptop=='False':
    constants.RESULTS = sys.argv[2]
    number = str(sys.argv[1])
    constants.RESULTS = constants.RESULTS + number
    config_file =  sys.argv[3]
    noise = str(sys.argv[4])
else:
    constants.RESULTS = 'Results_simulation_REALTEMP_'
    constants.RESULTS = constants.RESULTS + '1'
    number = '1'
    config_file = 'configs/'
    noise = '0'


dir_results = 'test'
dir_sim = dir_results + '_sim'
result_txt = constants.RESULTS + '/results_'
optimal_rate = 0.4
test = constants.RESULTS
begin = time.time()

if not os.path.exists(constants.RESULTS):
    os.makedirs(constants.RESULTS)
    print("Directory ", constants.RESULTS, " Created ")
else:
    print("Directory ", constants.RESULTS, " already exists")


# intialise
result = {'Name': [],
          'N status': [],
          'H_SEE': [],
          'phi_SEE': [],
          'pi_SEE': [],
          'PZ_SEE': [],
          'n_SEE': [],
          }
result_df = pd.DataFrame(result)


for file in glob.glob(config_file + "/*.json"):
    file_name = re.split("/|.json", file)[1]

    with open(file) as json_data_file:
        data = json.load(json_data_file)

    K = data['structure']['K']
    S = data['structure']['S']
    I = data['structure']['I']

    if onLaptop == 'True':
        max_iter = np.int(data['sampling']['max_iter'] / 10)
        min_iter = np.int(data['sampling']['min_iter'] / 10)
        burn_in = np.int(data['sampling']['burn_in']/10)
        batch = np.int(data['sampling']['batch']/10)
        var_calculation = np.int(data['sampling']['min_iter'] * 0.09)
    else:
        max_iter = np.int(data['sampling']['max_iter'])
        min_iter = np.int(data['sampling']['min_iter'])
        burn_in = np.int(data['sampling']['burn_in'])
        batch = np.int(data['sampling']['batch'])
        var_calculation = np.int(data['sampling']['min_iter']*0.9)

    phi_gamma = np.array(data['Gamma']['phi_gamma'])
    # F could be None, In that case, it will be generated using dirichlet distribution
    F_epsilon = np.tile(data['Gamma']['F_epsilon'], (K, 1))
    F_fraction = data['Gamma']['F_fraction']

    F = np.tile(data['Gamma']['F'], (K, 1))  # np.array([[9,2],[9,2],[9,2],[9,2],[9,2],[9,2]])

    gamma = data['theta']['gamma']
    gamma_sampling = data['theta']['gamma_sampling']
    theta_variable = data['theta']['theta_variable']

    sample_1 = pickle.load(open(f'{dir_sim}/{number}/sample_{file_name}', 'rb'))

    if noise=='0':
        n_lambda_tum = sample_1.n
    else:
        b = np.random.binomial(n=1,p=0.5,size=S)
        noise_pois = np.random.poisson(lam=int(noise),size=S)
        n_lambda_tum = sample_1.n+noise_pois*b+noise_pois*(1-b)



    tum_objs = []
    if os.path.exists(constants.RESULTS+'/'+file_name+'_chain_'+str(constants.CHAINS)):
        tum_all = [pickle.load(open(constants.RESULTS+'/'+file_name+'_chain_'+str(constants.CHAINS), 'rb'))]
    else:
        for cc in range(constants.CHAINS):
            tum_objs.append(
                tum.tumoroscope(name=constants.RESULTS+'/'+file_name+'_chain_'+str(cc),
                            K=sample_1.K, S=sample_1.S, r=None, p=None,
                            I=sample_1.I, avarage_clone_in_spot=sample_1.avarage_clone_in_spot, F=F,
                            C=sample_1.C, A=sample_1.A, D=sample_1.D, F_epsilon=F_epsilon, 
                            optimal_rate=optimal_rate, n_lambda=n_lambda_tum, gamma=gamma, pi_2D=pi_2D,
                            result_txt=result_txt + file_name + '.txt'))

        args_map = tum_objs

        with Pool(constants.CORES) as pool:
            tum_all = pool.map(run_in_parallel, args_map)
            # result_txt = result_txt+str(pool.name)

    logliks = [c.last_loglik for c in tum_all]
    tum_best = tum_all[np.argmax(logliks)]
    tum_best.H = 0
    pickle.dump(tum_best, open(f'{constants.RESULTS}/best_oo{constants.CHAINS}_{file_name}', 'wb'))

    if constants.CHAINS > 1:
        dir_out = dir_results + '/Results_plots_only_tum'
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        fig, axes = plt.subplots(3, constants.CHAINS, figsize=(15, 12))
        for c in range(constants.CHAINS):
            t = tum_all[c]
            axes[0, c].scatter(sample_1.H, t.inferred_H)
            axes[0, c].set_xlabel('True H')
            axes[0, c].set_ylabel('Inferred H')
            axes[0, c].set_title('log_likelihood = ' + str(np.round(t.last_loglik, 2)))
            axes[1, c].scatter(sample_1.phi, t.inferred_phi)
            axes[1, c].set_xlabel('True phi')
            axes[1, c].set_ylabel('Inferred phi')
            axes[2, c].scatter(sample_1.n, t.inferred_n)
            axes[2, c].set_xlabel('True N')
            axes[2, c].set_ylabel('Inferred N')
        plt.tight_layout()
        plt.savefig(f'{dir_out}/inference_{file_name}_{number}.png', dpi=200, bbox_inches='tight')
        plt.close()

    dir_out = dir_results + '/Results_numpy_only_tum'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    np.save(f'{dir_out}/inferred_H_{file_name}_{number}.npy', tum_best.inferred_H)
    np.save(f'{dir_out}/inferred_phi_{file_name}_{number}.npy', tum_best.inferred_phi)
    np.save(f'{dir_out}/inferred_N_{file_name}_{number}.npy', tum_best.inferred_n)
    np.save(f'{dir_out}/acceptance_rate_G_{file_name}_{number}.npy', tum_best.decision_matrix_G/tum_best.max_iter)
    np.save(f'{dir_out}/acceptance_rate_phi_{file_name}_{number}.npy', tum_best.decision_matrix_phi/tum_best.max_iter)
    np.save(f'{dir_out}/acceptance_rate_N_{file_name}_{number}.npy', tum_best.decision_matrix_n/tum_best.max_iter)
   

    best_dict = {'Name': file_name,
                    'N status': 'Variable',
                    'H_SEE': tum_best.H_SEE,
                    'phi_SEE': tum_best.phi_SEE,
                    'pi_SEE': tum_best.pi_SEE,
                    'PZ_SEE': tum_best.PZ_SEE,
                    'n_SEE': tum_best.n_SEE
                    }

    result_df = result_df.append(best_dict, ignore_index=True)


print(result_df)
dir_out = dir_results + '/csv_only_tum'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

result_df.to_csv(f'{dir_out}/results_{number}.csv')

