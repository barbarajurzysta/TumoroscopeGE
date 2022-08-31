import visualization as vis
import tumoroscope as tum
import pickle
import numpy as np
import random
import os
import pandas as pd
import json
import sys
import distutils.util
import seaborn as sns; sns.set_theme(color_codes=True)
import pre_processing

def main():
    genes_file = 'most_var_genes1000.txt'
    plot_colors = ['b','g','r','c','m','y','k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    col = ("#efefef",'#003f5c' ,'#AAAAAA' ,'#bc5090' ,'#ff6361', '#ffa600' ,"#47B39C","#74BBFB")

    ######  if data set change, it should change
    #sections_n_file = ['P1.2', 'P2.4', 'P3.3']
    #tree_clones = ['C0','C1','C2','C3']

    config = sys.argv[2]
    result_dir = sys.argv[1]
    move_dir = ''

    run_sampling = bool(distutils.util.strtobool(sys.argv[3]))
    save_observed = bool(distutils.util.strtobool(sys.argv[4]))
    vis_observed = bool(distutils.util.strtobool(sys.argv[5]))
    vis_results = bool(distutils.util.strtobool(sys.argv[6]))
    saved_inputs = bool(distutils.util.strtobool(sys.argv[7]))
    seed = int(sys.argv[8])

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print("Directory ", result_dir, " Created ")


    # file_log = open(result_dir + '/log.txt', 'w')
    # sys.stdout = file_log

    print('The config file is: ' + config)
    print('seed:', seed)
    with open(config) as json_data_file:
        data = json.load(json_data_file)
    ### making st data structure
    ### can be change by changing the data
    sections_n_file = []
    for name,value in data['observed'].items():
        sections_n_file.append(name)

    def files_to_inputs(data,save_observed):
        st_sections = [[]]*len(sections_n_file)
        counter = 0
        for sections in sections_n_file:
            st_sections[counter] = pd.read_csv(move_dir + data['observed'][sections], delimiter = "\t", dtype={"refContig":"string","refPos":int,"refAllele":"string","base":"string","source":"string", "spot":"string", "cnt":int})
            st_sections[counter]['source'] = sections
            st_sections[counter]['spot'] = sections+'_' + st_sections[counter]['spot']
            counter = counter + 1
        frames = st_sections

        ###########################################################################################################
        ###########################################################################################################


        #barcode = pd.read_csv(data['structure']['barcode'], delimiter = "\t",dtype={"barcode": "string", "x": int,"y": int})



        print("######################  pre processing started: preparing observed variables ####################")




        st_wes = pre_processing.generate_st_wes(frames, move_dir + data['C_variation']['WES_file'],data['criteria']['offset'])
        inputs_dump_dir = move_dir + data['results']['data_tumoroscope_inputs']
        pickle.dump(st_wes, open(inputs_dump_dir + 'st_wes.pickle', 'wb'))


        #df of selected spots

        n_s_data = pd.read_csv(move_dir + data['n_variation']['n_file'],sep=',')
        n_s_data = n_s_data[~n_s_data.isin([np.nan, np.inf, -np.inf]).any(1)]
        n_s_data = n_s_data.drop_duplicates()
        n_s_data = n_s_data[n_s_data['type'].str.contains('Cancer')]
        if(n_s_data.barcode.duplicated().any()==True):
            n_s_data[n_s_data.barcode.duplicated()]=np.nan
            n_s_data = n_s_data[~n_s_data.isin([np.nan, np.inf, -np.inf]).any(1)]
            print("There are duplications in the barcode(probably different types)")
        #selected_spots = pre_processing.n_barcode_merge(data['structure']['selected_spot_file'],sections_n_file,barcode)

        st_wes_selected_spots=st_wes[st_wes['spot'].isin(n_s_data['barcode'])]


        mutations_locations = np.unique(st_wes_selected_spots['gene'])
        mutations_locations_minus1 = np.unique(st_wes_selected_spots['gene_minus1'])
        # the rows are mutation_id and the columns are clones for C_ik
        K, C_ik=pre_processing.generate_C(data['C_variation']['method'], mutations_locations, move_dir + data['C_variation']['phyloWGS'], data['C_variation']['tree'],move_dir + data['C_variation']['selected_canopy_tree'])
        if np.sum(np.sum(C_ik>1)):
            C_ik[C_ik>1] = 1
            print("Warning! C has numbers higher than 1")
        F_epsilon,F = pre_processing.generate_F(data['Gamma']['F_epsilon'],move_dir + data['Gamma']['F_file'],K)




        #if data['structure']['all_spots']==True:
        #    n_s_data,spots_order,S,n_lambda,n = pre_processing.generate_n_lambda(data['n_variation']['n_file'],sections_n_file,barcode,data['n_variation']['n_sampling'])
        #    A,D,A_df,D_df = pre_processing.generate_A_D(st_wes,offset,spots_order)
        #else:
            #n_s_data = pd.read_csv(data['n_variation']['n_file'],sep=',')


        #S = len(n_s_data['barcode'])
        #spots_order = n_s_data['barcode']
        #n_lambda = n_s_data.nuclei

            #n_s_data,spots_order,S,n_lambda,n = pre_processing.generate_n_lambda(data['structure']['selected_spot_file'],sections_n_file,barcode,data['n_variation']['n_sampling'])


        A,D,A_df,D_df = pre_processing.generate_A_D(st_wes_selected_spots,data['criteria']['offset'],n_s_data['barcode'])



        #spots_order,S,n_lambda = pre_processing.update_spots_lambda(n_s_data)


        if np.sum(n_s_data['barcode'].isin(A.columns))!=len(n_s_data['barcode']):
            Warning('Out of '+str(len(n_s_data['barcode']))+' spots in the cell number file, '+str(
                np.sum(n_s_data['barcode'].isin(A.columns)))+' number of them showed up in the st file.')
            Warning('Out of ' + str(len(C_ik.index)) + ' mutations in the tree, ' + str(
                np.sum(C_ik.index.isin(A.index))) + ' number of them showed up in the st file.')
            #n_lambda.reindex(spots_order.index)
            #n_lambda = n_lambda[spots_order.index]
            C_ik = C_ik[C_ik.index.isin(A.index)]
            C_ik = C_ik[np.sum(C_ik,axis=1)>0]
            #I = len(C_ik)
            #S = len(A.columns)
            A=A[A.index.isin(C_ik.index)]
            D=D[D.index.isin(C_ik.index)]

            C_ik = C_ik.reindex(A.index)

            n_s_data = n_s_data[n_s_data['barcode'].isin(A.columns)]
            n_s_data = n_s_data.reset_index(drop=True)
            #n_s_data = n_s_data[n_s_data['barcode'].isin(A.columns)]
            #spots_order, S, n_lambda = pre_processing.update_spots_lambda(n_s_data)

        exs = []
        for section in sections_n_file:
            expr_data = pd.read_csv(move_dir + data['expression'][section], delimiter="\t", index_col=0)
            in_section = n_s_data[n_s_data['section']==section]
            coordinates = expr_data.columns.intersection(in_section['coordinates'])
            expr_data = expr_data[coordinates]
            d = {c:b for c, b in zip(in_section.coordinates, in_section.barcode)}
            expr_data = expr_data.rename(columns=d)
            exs.append(expr_data)
        expr_data = pd.concat(exs, axis=1).dropna()
        expr_data = expr_data[expr_data.sum(axis=1)>0]   # excluding genes with expression = 0
        expr_data = expr_data
        n_s_data = n_s_data[n_s_data['barcode'].isin(expr_data.columns)]
        expr_data = expr_data[n_s_data.barcode]   # setting in right order
        with open(genes_file) as f:
            genes = f.readlines()
            genes = [gene.strip() for gene in genes]
        expr_data = expr_data.loc[genes]
        sections_array = np.array(expr_data.columns.str[:4])
        Y = np.array(expr_data)
        Y = Y.T

        A = A[n_s_data.barcode]
        D = D[n_s_data.barcode]

        D = D[A.sum(axis=1)>data['criteria']['st_read_limit']]
        A = A[A.sum(axis=1)>data['criteria']['st_read_limit']]
        C_ik = C_ik[C_ik.index.isin(A.index)]


        if save_observed==True:
            lambda_file = open(result_dir+"/n_lambda.txt", "w")
            np.savetxt(lambda_file, n_s_data['nuclei'],fmt='%s')
            a_file = open(result_dir+"/A.txt", "w")
            np.savetxt(a_file, A,fmt='%s')
            d_file = open(result_dir+"/D.txt", "w")
            np.savetxt(d_file, D,fmt='%s')
            s_file = open(result_dir+"/spots_order.txt", "w")
            np.savetxt(s_file, n_s_data['barcode'], fmt='%s')
            m_file = open(result_dir+"/mutations_order.txt", "w")
            C_ik.to_csv(result_dir+'/C_ik.csv', index=False,sep='\t')
            np.savetxt(m_file, np.unique(A_df["i"]), fmt='%s')
            expr_data.to_csv(result_dir+'/expr.csv', index=True,sep='\t')

        n_s_data['nuclei'][n_s_data['nuclei']==0]=1

        A_df = A_df[A_df['s'].isin(A.columns)]
        A_df = A_df[A_df['i'].isin(A.index)]
        D_df = D_df[D_df['s'].isin(D.columns)]
        D_df = D_df[D_df['i'].isin(D.index)]

        pickle.dump(A_df, open(inputs_dump_dir + 'A_df.pickle', 'wb'))
        pickle.dump(D_df, open(inputs_dump_dir + 'D_df.pickle', 'wb'))
        pickle.dump(F, open(inputs_dump_dir + 'F.pickle', 'wb'))
        pickle.dump(F_epsilon, open(inputs_dump_dir + 'F_epsilon.pickle', 'wb'))
        pickle.dump(A, open(inputs_dump_dir + 'A.pickle', 'wb'))
        pickle.dump(D, open(inputs_dump_dir + 'D.pickle', 'wb'))
        pickle.dump(n_s_data, open(inputs_dump_dir + 'n_s_data_final.pickle', 'wb'))
        pickle.dump(C_ik, open(inputs_dump_dir + 'C_ik.pickle', 'wb'))
        pickle.dump(Y, open(inputs_dump_dir + 'Y.pickle', 'wb'))

        textfile = open(result_dir + '/spots_order.txt', "w")
        for element in A.columns:
            textfile.write(element + "\n")
        textfile.close()

        textfile = open(result_dir + '/mutations_order.txt', "w")
        for element in A.index:
            textfile.write(element + "\n")
        textfile.close()

        textfile = open(result_dir + '/genes_order.txt', "w")
        for element in expr_data.index:
            textfile.write(element + "\n")
        textfile.close()

        return F, F_epsilon, A_df, D_df, C_ik, A, D, n_s_data, Y, sections_array


    # def read_saved_files_to_input(data):
    #     A_df=pickle.load(open(inputs_dump_dir + 'A_df.pickle', 'rb'))
    #     D_df=pickle.load( open(inputs_dump_dir + 'D_df.pickle', 'rb'))
    #     A=pickle.load( open(inputs_dump_dir + 'A.pickle', 'rb'))
    #     D=pickle.load( open(inputs_dump_dir + 'D.pickle', 'rb'))
    #     n_s_data=pickle.load(open(inputs_dump_dir + 'n_s_data_final.pickle', 'rb'))
    #     F=pickle.load( open(inputs_dump_dir + 'F.pickle', 'rb'))
    #     F_epsilon=pickle.load( open(inputs_dump_dir + 'F_epsilon.pickle', 'rb'))
    #     C_ik = pickle.load(open(inputs_dump_dir + 'C_ik.pickle', 'rb'))
    #     Y = pickle.load(open(inputs_dump_dir + 'Y.pickle', 'rb'))
    #     return F, F_epsilon, A_df, D_df, C_ik, A, D, n_s_data, Y

    if saved_inputs==True:
        F, F_epsilon, A_df, D_df, C_ik, A, D, n_s_data, Y = read_saved_files_to_input(data)

    else:
        F, F_epsilon, A_df, D_df, C_ik, A, D, n_s_data, Y, sections_array = files_to_inputs(data,save_observed)
        print("inputs saved!")

    visualization_dir = result_dir+'/visualization_'+data['structure']['section']
    vis_1 = vis.visualization(visualization_dir)
    if vis_observed == True:
        print("###################### Visualization of the observed variables  ####################")
        vis_1.plot_F_gamma(F_epsilon, F, plot_colors)
        vis_1.hist_matrix(A_df['value'].values.flatten(), 50, 'A')
        vis_1.hist_matrix(D_df['value'].values.flatten(), 50, 'D')
        vis_1.gamma(np.array(data['Gamma']['phi_gamma'])[0], np.array(data['Gamma']['phi_gamma'])[1], 'Phi')
        vis_1.heatmap_seaborn(C_ik.to_numpy(), 'C_seaborn', 'clones', 'mutations', False, 0.5)
        g = sns.clustermap(C_ik, cmap="Blues")
        g.ax_row_dendrogram.set_xlim([0, 0])
        g.savefig(result_dir + "/C_ik_clustered.png")


    def estimate_parameters(Y, n_lambda, sections_array):
        sections = list(set(sections_array))
        means = [np.mean(Y[sections_array==section]) for section in sections]
        means = np.array(means)
        scaling_factors = np.mean(means) / means
        scaling_factors_array = np.zeros(len(sections_array))
        for section, scaling_factor  in zip(sections, scaling_factors):
            scaling_factors_array[sections_array==section] = scaling_factor
        scaling_factors_array = scaling_factors_array.reshape(-1,1)
        Y = Y * scaling_factors_array
        g = Y.shape[1]
        pred_p = np.mean(Y, axis=0) / np.var(Y, axis=0)
#        pred_p = np.zeros(g) + np.median(pred_p)  # common p seems to work better
        pred_B = []
        for i in range(g):
            pred_Bg = np.mean(Y[:,i]/n_lambda)
            pred_B.append(pred_Bg)
        pred_beta = np.var(pred_B) / np.mean(pred_B) * 2   # *2 because this method of estimation probably results in too low beta
        pred_alpha = pred_B / pred_beta
        return pred_p, pred_alpha, pred_beta, 1 / scaling_factors_array

    result_txt = result_dir + '/' + data['results']['text_result'] + data['structure']['section'] + '.txt'
    result_obj = result_dir + '/' + data['results']['text_result'] + data['structure']['section']

    print("###################### Start of the sampling  ####################")
    if run_sampling is True:
        # name, K, S, g, r, p, I, avarage_clone_in_spot, F, C, A, D, F_epsilon, n, optimal_rate, n_lambda,
        # gamma, pi_2D, result_txt, Y, p_y, b_alpha, b_beta

        # 'Y', 'p_y', 'b_alpha', and 'b_beta'
        K = len(C_ik.columns)
        S = len(n_s_data['barcode'])
        g = Y.shape[1]
        print(f'running on {K} clones, {S} spots and {g} genes')
        pred_p, pred_alpha, pred_beta, pred_t = estimate_parameters(Y, n_s_data['nuclei'].astype(float), sections_array)
        print('pred_p:', np.min(pred_p), '-', np.max(pred_p))
        print('pred_alpha:', np.min(pred_alpha), '-', np.max(pred_alpha))
        print('pred_beta:', pred_beta)

# r=np.array(data['Gamma']['phi_gamma'])[0], p=np.array(data['Gamma']['phi_gamma'])[1]
        r, p = 0.005, 2
        tum_1 = tum.tumoroscope(name=result_obj, K=K, S=S, g=g, r=r,
                                p=p, I=len(C_ik),
                                avarage_clone_in_spot=data['Z_variation']['avarage_clone_in_spot'], F=F,
                                C=C_ik.to_numpy(), A=A.to_numpy(), D=D.to_numpy(), F_epsilon=F_epsilon,
                                optimal_rate=data['structure']['optimal_rate'], n_lambda=n_s_data['nuclei'].astype(float),
                                pi_2D=data['structure']['pi_2D'],result_txt=result_txt,
                                Y=Y, p_y=pred_p, b_alpha=pred_alpha, b_beta=pred_beta, t=pred_t)

        tum_1.gibbs_sampling(seed=seed, min_iter=np.int(data['sampling']['min_iter']),
                             max_iter=np.int(data['sampling']['max_iter']), batch=np.int(data['sampling']['batch']),
                             simulated_data=None, n_sampling=data['n_variation']['n_sampling'],
                             F_fraction=data['Gamma']['F_fraction'], pi_2D=data['structure']['pi_2D'],
                             th=data['Z_variation']['threshold'], every_n_sample=data['sampling']['every_n_sample'],
                             changes_batch=data['sampling']['changes_batch'])
        #vis_1.visualizing_inferred_variables(tum_1)
        pickle.dump(tum_1, open(result_obj, 'wb'))
    else:
        print("loading existing object")
        tum_1 = pickle.load(open(result_obj, 'rb'))

    #vis_1.heatmap_seaborn(C_ik.to_numpy(), 'C_seaborn', 'clones', 'mutations', False, 0.5)

    def post_tumoroscope(K,tum_1,spots_order,result_dir,section,visualization_dir,n_s_data,n_lambda,col,vis_1,sections_n_file):

        print("###################### saving and visualizing the results  ####################")
        tree_clones = [f"C{i}" for i in range(1,(K+1))]
        inferred_h = pd.DataFrame(data=tum_1.inferred_H, index=spots_order, columns=tree_clones).round(3)
        inferred_z = pd.DataFrame(data=tum_1.inferred_Z, index=spots_order, columns=tree_clones).round(3)
        inferred_p_z = pd.DataFrame(data=tum_1.inferred_P_Z, index=spots_order, columns=tree_clones).round(3)
        inferred_n = pd.DataFrame(data=tum_1.inferred_n, index=spots_order, columns=['n']).round(3)

        inferred_h.to_csv(result_dir + '/' + section + '_h.txt', header=tree_clones, sep='\t', mode='w')
        inferred_z.to_csv(result_dir + '/' + section + '_z.txt', header=tree_clones, sep='\t', mode='w')
        inferred_n.to_csv(result_dir + '/' + section + '_n.txt', header=[section], sep='\t', mode='w')

        pre_processing.plot_prior_inferred_n(n_lambda,tum_1.inferred_n,section,visualization_dir)

        dir_out = 'rerun/Results_numpy_known_params'
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        np.save(result_dir + '/' + section + '_h.npy', tum_1.inferred_H)
        np.save(result_dir + '/' + section + '_b.npy', tum_1.inferred_B)
        np.save(result_dir + '/' + section + '_n.npy', tum_1.inferred_n)

        n_s_data_h = pd.merge(left=n_s_data, right=inferred_h, how="left", left_on=['barcode'], right_on=['barcode'])
        n_s_data_z = pd.merge(left=n_s_data, right=inferred_z, how="left", left_on=['barcode'], right_on=['barcode'])

        for n_section in sections_n_file:
            inferred_H_section = n_s_data_h[n_s_data_h['section'] == n_section][tree_clones].reset_index(drop=True)
            inferred_Z_section = n_s_data_z[n_s_data_z['section'] == n_section][tree_clones].reset_index(drop=True)
            n_s_data_temp = n_s_data[n_s_data['section'] == n_section].reset_index(drop=True)
            vis_1.plot_pie_chart_H(n_s_data_temp, inferred_H_section, col, n_section, tree_clones)
            vis_1.plot_pie_chart_HZ(n_s_data_temp, inferred_H_section, inferred_Z_section, col, n_section, tree_clones)
            vis_1.plot_spots_each_clone( n_s_data_temp,inferred_H_section , n_section)

        vis_1.visualizing_inferred_variables(tum_1)

    if vis_results==True:
        n_s_data['x'] = n_s_data['x'].astype(float)
        n_s_data['y'] = n_s_data['y'].astype(float)
        n_s_data['nuclei'] = n_s_data['nuclei'].astype(float)
        post_tumoroscope(len(C_ik.columns),tum_1,n_s_data['barcode'],result_dir,data['structure']['section'],visualization_dir,n_s_data,n_s_data['nuclei'],col,vis_1,sections_n_file)

    # file_log.close()

if __name__=='__main__':
    main()
