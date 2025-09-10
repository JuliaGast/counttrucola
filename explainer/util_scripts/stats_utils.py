"""
Code from @article{huang2024tgb2,
  title={TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs},
  author={Gastinger, Julia and Huang, Shenyang and Galkin, Mikhail and Loghmani, Erfan and Parviz, Ali and Poursafaei, Farimah and Danovitch, Jacob and Rossi, Emanuele and Koutis, Ioannis and Stuckenschmidt, Heiner and      Rabbany, Reihaneh and Rabusseau, Guillaume},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
https://github.com/JuliaGast/TGB2/tree/main/stats_figures


This script computes statistics for all datasets in TGB2.
Basically everything that we report in the paper table, as well as some additional statistics like number of edges per timestep
Needed:
datasets in dataset folder (no preprocessing needed)
Output: 
dataset_stats.csv # statistics for a  datasets - stored in the respective dataset folder
numedges_datasetname.json # number of edges per timestep (to create the figures)
"""

## imports
import numpy as np
import os
import json
import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta


#internal imports 

import util_scripts.dataset_utils as du


def compute_stats(dataset_name, dataset):
    occ_threshold = 5
    relations = dataset.relations
    num_rels = dataset.num_rels
    num_rels_without_inv = dataset.num_rels_half


    subjects = dataset.subjects
    objects= dataset.objects
    num_nodes = dataset.dataset.num_nodes 
    timestamps_orig = dataset.timestamps_orig 
    timestamps = dataset.timestamps
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # csv_dir = os.path.join( current_dir, dataset_name)
    # np.savetxt(csv_dir +"/"+dataset_name+"timestamps.csv", timestamps,fmt='%i', delimiter=",")
    all_quads = np.stack((subjects, relations, objects, timestamps, timestamps_orig), axis=1)
    train_data = dataset.train_data
    val_data = dataset.val_data
    test_data = dataset.test_data


    first_ts = timestamps_orig[0]
    last_ts = timestamps_orig[-1]

    last_ts_converted = timestamps[-1]
    first_ts_converted = timestamps[0]

    # timestamp strings for figure (first and last timestamp in dataset)
    if 'wikidata' in dataset_name or 'smallpedia' in dataset_name or 'yago' in dataset_name or 'wikiold' in dataset_name or 'gdelt' in dataset_name:
        first_ts_string = str(first_ts)
        last_ts_string = str(last_ts)
    elif 'icews14' in dataset_name:
        first_ts_string = '2014-01-01'
        first_ts_dt = datetime.strptime(first_ts_string, '%Y-%m-%d')
        last_ts_string = (first_ts_dt + timedelta(days=int(last_ts_converted - first_ts_converted))).strftime('%Y-%m-%d')
    elif 'icews18' in dataset_name:
        first_ts_string = '2018-01-01'
        first_ts_dt = datetime.strptime(first_ts_string, '%Y-%m-%d')
        last_ts_string = (first_ts_dt + timedelta(days=int(last_ts_converted - first_ts_converted))).strftime('%Y-%m-%d')


    elif 'thgl' in dataset_name:
        first_ts_string = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        last_ts_string = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    else:
        first_ts_string = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime('%Y-%m-%d')
        last_ts_string = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime('%Y-%m-%d')

    print(dataset_name, "first timestamp:", first_ts_string, "last timestamp:", last_ts_string)
    

    # compute number of quads in train/val/test set
    num_train_quads = train_data.shape[0]
    num_val_quads = val_data.shape[0]
    num_test_quads = test_data.shape[0]
    num_all_quads = num_train_quads + num_val_quads + num_test_quads
    print(num_all_quads)

    # compute inductive nodes
    test_ind_nodes = du.num_nodes_not_in_train(train_data, test_data)
    val_ind_nodes = du.num_nodes_not_in_train(train_data, val_data)
    test_ind_nodes_perc = test_ind_nodes/num_nodes
    val_ind_nodes_perc = val_ind_nodes/num_nodes

    # compute number of timesteps in train/val/test set
    num_train_timesteps = len(np.unique(train_data[:,-1]))
    num_val_timesteps = len(np.unique(val_data[:,-1]))
    num_test_timesteps = len(np.unique(test_data[:,-1]))
    num_all_ts = num_train_timesteps + num_val_timesteps + num_test_timesteps

    # compute number on nodes in valid set or test set that have not been seen in train set
    # compute recurrency degree
    # compute average duration of facts
    timestep_range = 1+np.max(timestamps) - np.min(timestamps)

    ts_all = du.TripleSet()
    ts_all.add_triples(all_quads, num_rels_without_inv, timestep_range)
    ts_all.compute_stat()
    ts_test = du.TripleSet()
    ts_test.add_triples(test_data, num_rels_without_inv, timestep_range)
    ts_test.compute_stat()

    lens = []
    for timesteps in ts_all.timestep_lists:
        lens.append(len(timesteps))

    count_previous = 0
    count_sometime = 0
    count_all = 0
    for qtriple in ts_test.triples:    
        (s,r,o,t) = qtriple
        k = ts_all.get_latest_ts(s,r,o, t)
        count_all += 1
        if k + 1 == t: count_previous += 1
        if k > -1 and k < t: count_sometime += 1

    print("DATATSET:  " + dataset_name)
    print("all:       " +  str(count_all))
    print("previous:  " +  str(count_previous))
    print("sometime:  " +  str(count_sometime))
    print("f-direct (DRec):   " +  str(count_previous / count_all))
    print("f-sometime (Rec): " +  str(count_sometime / count_all))

    print(f"the mean number of timesteps that a triple appears in is {np.mean(lens)}")
    print(f"the median number of timesteps that a triple appears in is {np.median(lens)}")
    print(f"the maximum number of timesteps that a triple appears in is {np.max(lens)}")

    # Compute max consecutive timesteps per triple
    results = [du.max_consecutive_numbers(inner_list) for inner_list in ts_all.timestep_lists]
    print(f"number of timesteps is {ts_all.num_timesteps}")
    print(f"number of total triples is {ts_all.num_triples}")
    print(f"number of distinct triples is {len(ts_all.timestep_lists)}")
    print(f"the mean max number of 100*consecutive timesteps/number of timesteps that a triple appears in is {100*np.mean(results)/ts_all.num_timesteps}")
    print(f"the median max number of 100*consecutive timesteps/number of timesteps that a triple appears in is {100*np.median(results)/ts_all.num_timesteps}")
    print(f"the maximum max number of 100*consecutive timesteps/number of timesteps that a triple appears in is {100*np.max(results)/ts_all.num_timesteps}")
    print(f"the mean max number of consecutive timesteps that a triple appears in is {np.mean(results)}")
    print(f"the median max number of consecutive timesteps that a triple appears in is {np.median(results)}")
    print(f"the maximum max number of consecutive timesteps that a triple appears in is {np.max(results)}")
    print(f"the std for max number of consecutive timesteps that a triple appears in is {np.std(results)}")

    direct_recurrency_degree = count_previous / count_all
    recurrency_degree = count_sometime / count_all
    consecutiveness_degree =  np.mean(results) # the mean max number of consecutive timesteps that a triple appears in


    # compute number of triples per timestep
    n_nodes_list = []
    n_edges_list = []

    ts_set = list(set(timestamps_orig))
    ts_set.sort()

    if 'tkg' in dataset_name:
        all_possible_orig_timestamps = dataset.timestamps_orig 

    no_nodes_list = []
    no_nodes_list_orig = []
    no_nodes_datetime = []
    for t in ts_all.t_2_triple.keys():
        num_nodes_ts = len(ts_all.unique_nodes(ts_all.t_2_triple[t]))
        n_nodes_list.append(num_nodes_ts)
        n_edges_list.append(len(ts_all.t_2_triple[t]))
        if 'tkg' in dataset_name:
            if num_nodes_ts == 0:
                if t not in no_nodes_list:
                    no_nodes_list.append(t)
                    no_nodes_list_orig.append(all_possible_orig_timestamps[t])
                    no_nodes_datetime.append(datetime.fromtimestamp(all_possible_orig_timestamps[t]).strftime('%Y-%m-%d'))
    # compute seasonality of num nodes over time: 
    seasonal_value =1
    seasonal_value = du.estimate_seasons(n_nodes_list)
    if seasonal_value == 1:
        print('there was no seasonality for number of nodes found')
    else:
        print(f'the seasonality for number of nodes is {seasonal_value}')
    if 'tkgl' in dataset_name:
        print('we have 0 nodes for' + str(len(no_nodes_list)) + ' timesteps')
        print('0 nodes for timesteps: ', no_nodes_list)
        print('this is original unix timestamps: ', no_nodes_list_orig)
        print('this is datetime: ', no_nodes_datetime)
    else:
        print('we have 0 nodes for' + str(len(no_nodes_list)) + ' timesteps')

            
    print(f"average number of triples per ts is {np.mean(n_edges_list)}")
    print(f"std for average number of triples per ts is {np.std(n_edges_list)}")
    print(f"min/max number of triples per ts is {np.min(n_edges_list), np.max(n_edges_list)}")

    print(f"average number of nodes per ts is {np.mean(n_nodes_list)}")
    print(f"std for average number of nodes per ts is {np.std(n_nodes_list)}")
    print(f"min/max number of nodes per ts is {np.min(n_nodes_list), np.max(n_nodes_list)}")

    # create a dict with number of endges and number of 
    to_be_saved_dict = {}
    to_be_saved_dict['num_triples'] = n_edges_list
    to_be_saved_dict['num_nodes'] = n_nodes_list

    figs_dir = os.path.join('..', 'files', 'analysis', dataset_name)


    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
     # save the number of edges per timestep in a json file (to create the figures later)
    save_path = (os.path.join(figs_dir,f"numedges_{dataset_name}.json")) 
    save_file = open(save_path, "w")
    json.dump(to_be_saved_dict, save_file)
    save_file.close()

    
    # Save stats_dict as CSV
    save_path = (os.path.join(figs_dir, "dataset_stats.csv"))
    # save the statistics in a dictionary
    stats_df = du.create_dict_and_save(dataset_name, num_rels_without_inv, num_nodes, num_train_quads, num_val_quads, num_test_quads, 
                        num_all_quads, num_train_timesteps, num_val_timesteps, num_test_timesteps, num_all_ts,
                        test_ind_nodes, test_ind_nodes_perc, val_ind_nodes, val_ind_nodes_perc, 
                        direct_recurrency_degree, recurrency_degree, consecutiveness_degree,
                        np.mean(n_edges_list), np.std(n_edges_list), np.min(n_edges_list), np.max(n_edges_list),
                        np.mean(n_nodes_list), np.std(n_nodes_list), np.min(n_nodes_list), np.max(n_nodes_list),
                        seasonal_value, first_ts_string, last_ts_string, save_path)
    


    # Read the CSV file into a DataFrame

    # create dictionaries that contain different combis of key:rel, values:[s,o,t] and so on
    timestep_range = 1+np.max(timestamps) - np.min(timestamps)
    all_possible_timestep_indices = [i for i in range(timestep_range)]
    ts_all = du.TripleSet()
    ts_all.add_triples(all_quads, num_rels_without_inv, timestep_range)
    ts_all.compute_stat()
    ts_test = du.TripleSet()
    ts_test.add_triples(test_data, num_rels_without_inv, timestep_range)
    ts_test.compute_stat()

    ############################## Compute Stats ##############################
    # compute the number of ocurances of each relation
    rels_occurences ={}
    for rel in ts_all.r_2_triple.keys():
        rels_occurences[rel] = len(ts_all.r_2_triple[rel])

    # Sort the dictionary by values in descending order
    sorted_dict = dict(sorted(rels_occurences.items(), key=lambda item: item[1], reverse=True))

    # Take the top k key-value pairs and sum up their values
    top_k = dict(list(sorted_dict.items())[:k]) # highest k relations
    bad_k = dict(list(sorted_dict.items())[-k:]) # lowest k relations
    plot_names = du.set_plot_names(top_k, sorted_dict, dataset_name, dataset.rels_id_to_string ) #names to be included in the plot  
     
    num_occurences_dict = {}
    mean_std_max_min_dict = {}
    high_occurences = {}
    low_occurences = {}
    done_dict ={}
    # compute the mean, std, max, min, median of the number of occurences of triples for each relation
    for rel in ts_all.r_2_triple.keys():
        num_occurences_dict[rel] = []
        for triple in ts_all.r_2_triple[rel]:
            (s,r,o,t) = triple[0], rel, triple[1], triple[2]
            if (s,r,o) in done_dict.keys():
                continue
            else:
                done_dict[(s,r,o)] = 1
                count_num_occurences = len(ts_all.sub_rel_2_obj[s][r][o])
                num_occurences_dict[rel].append(count_num_occurences)
        mean_std_max_min_dict[rel] = (np.mean(num_occurences_dict[rel]), np.std(num_occurences_dict[rel]), np.max(num_occurences_dict[rel]), np.min(num_occurences_dict[rel]), np.median(num_occurences_dict[rel]))
        if mean_std_max_min_dict[rel][0] < occ_threshold:
            low_occurences[rel] = rels_occurences[rel]
        else:
            high_occurences[rel] = rels_occurences[rel]

    # compute for each relation the max consecutive timesteps of each triple
    lists_per_rel = {}
    mean_per_rel = {}
    for rel in ts_all.rel_sub_obj_t.keys():
        ts_lists = ts_all.create_timestep_lists(ts_all.rel_sub_obj_t[rel])
        for list_r in ts_lists:
            max_cn = du.max_consecutive_numbers(list_r)

            if rel not in lists_per_rel.keys():
                lists_per_rel[rel] = [max_cn]
            else:
                lists_per_rel[rel].append(max_cn)
        mean_per_rel[rel] = np.mean(lists_per_rel[rel])


    #only for the most prominent relations
    statistics_dict_prominent = {}
    for rel in top_k.keys():
        rel_key = dataset.rels_id_to_string[rel]
        # print(rel_key)
        statistics_dict_prominent[rel_key] = mean_std_max_min_dict[rel]


    ## create dataframe
    # each line in the dataframe is one relation
    # columns: [relation,  recurrency degree, direct recurrency degree, consecutiveness value,number of distinct triples,
    # number of total occurences,  mean_occurence per_triple, max_occurence per_triple, min_occurence per_triple, median_occurence per_triple]
    # df = pd.DataFrame(columns=['relation', 'rel_string_id', 'rel_string_word', 'recurrency_degree', 'direct_recurrency-degree', 'consecutiveness_value', \
    #                            'number_distinct_triples', 'number_total_occurences', 'mean_occurence_per_triple', \
    #                             'max_occurence_per_triple', 'min_occurence_per_triple', 'median_occurence_per_triple'])
    # for each relation: compute stats and add line to dataframe
    ## compute recurrency degree
    new_rows = []
    for rel in ts_all.r_2_triple.keys():


        rel_string_id = str(rel)
        word = dataset.rels_id_to_string[rel]
        if rel in ts_test.r_2_triple.keys():
            recurrency_degree, direct_recurrency_degree = du.compute_rec_drec(ts_test.r_2_triple[rel],rel, ts_all)
        else:
            recurrency_degree = 0
            direct_recurrency_degree = 0
            consecutiveness_value = 0
        consecutiveness_value = du.compute_consecutiveness(ts_all.rel_sub_obj_t[rel],ts_all) # TODO: implement this function
        number_distinct_triples = du.compute_number_distinct_triples(ts_all.rel_sub_obj_t[rel]) # TODO: implement this function
        number_total_occurences = len(ts_all.r_2_triple[rel]) # for how many triples does the relation occur in the dataset
        mean_occurence_per_triple = mean_std_max_min_dict[rel][0]
        max_occurence_per_triple = mean_std_max_min_dict[rel][2]
        min_occurence_per_triple = mean_std_max_min_dict[rel][3]
        median_occurence_per_triple = mean_std_max_min_dict[rel][4]

        data = {'relation': rel, 'rel_string_id':rel_string_id, 'rel_string_word':word, 
                        'recurrency_degree': recurrency_degree, 'direct_recurrency-degree': direct_recurrency_degree, 
                        'consecutiveness_value': consecutiveness_value, 
                        'number_distinct_triples': number_distinct_triples,
                        'number_total_occurences': number_total_occurences, 
                        'mean_occurence_per_triple': mean_occurence_per_triple, 
                        'max_occurence_per_triple': max_occurence_per_triple, 
                        'min_occurence_per_triple': min_occurence_per_triple, 
                        'median_occurence_per_triple': median_occurence_per_triple} 
        new_rows.append(data)

    df = pd.DataFrame(new_rows)
    # df = pd.concat([df, new_df], ignore_index=True)

    df_sorted = df.sort_values(by='number_total_occurences', ascending=False).reset_index(drop=True)


    # save dataframe to csv
    df_sorted.to_csv(os.path.join(figs_dir, f"relation_statistics_{dataset_name}.csv"), index=False)
    



    return stats_df, to_be_saved_dict, df_sorted