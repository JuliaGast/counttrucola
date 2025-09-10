import os
import pandas as pd
import datetime
import numpy as np

def sort_out_pathnames(options, dataset_name):
    """
    creates and returns the pathnames for all sorts of files we want to store or load from somewhere, e.g. the learn_data, results, rankings, rules, and stats files
    """
    rules_dir = os.path.join("files", "rules")
    if not os.path.isdir(rules_dir):
        os.makedirs(rules_dir)    

    path_rules, all_rule_types_false = get_path_rules_name(options, rules_dir, dataset_name)

    if options['LEARN_DATA_PATH'] == 'default':
        learn_data_dir = os.path.join("files", "learn_data")
    else:
        print("Using custom learn data path: ", options['LEARN_DATA_PATH'])
        learn_data_dir = options['LEARN_DATA_PATH']
    if not os.path.isdir(learn_data_dir):
        os.makedirs(learn_data_dir)
    if options["MULTI_FLAG"]:
        learn_data_path = os.path.join(learn_data_dir, dataset_name+"-multi-learn_data"+str(options["LEARN_WINDOW_SIZE"])+'_'+str(options["RR_OFFSET"])+"_.pkl")
    else:
        learn_data_path = os.path.join(learn_data_dir, dataset_name+"-learn_data"+str(options["LEARN_WINDOW_SIZE"])+'_'+str(options["RR_OFFSET"])+"_.pkl")
    
    if options['RANKINGS_PATH']== 'default':
        rankings_dir = os.path.join("files", "rankings")
    else:
        print("Using custom rankings path: ", options['RANKINGS_PATH'])
        rankings_dir = options['RANKINGS_PATH']
    if not os.path.isdir(rankings_dir):
            os.makedirs(rankings_dir)
    rankings_name = dataset_name + "-rankings{}.txt"

    results_dir = os.path.join("files", "results")
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    results_name = dataset_name + "-results.csv"
    results_path = os.path.join(results_dir, results_name)
    figure_path = os.path.join(results_dir, "figures")

    stats_dir = os.path.join("files", "stats")
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)
    stats_file_path = os.path.join(stats_dir, dataset_name+"-rulestats.txt")
    log_path = os.path.join(stats_dir, dataset_name+"-learnlog.txt" )

    # path_rankings_test = os.path.join(rankings_dir, "test_"+rankings_name) # this is the path where the computed rankings are written to
    path_rankings_test = get_path_rankings_name(rankings_dir, rankings_name, dataset_name, "test", options) # this is the path where the computed rankings are written to
    path_rankings_val = get_path_rankings_name(rankings_dir, rankings_name, dataset_name, "val", options) # this is the path where the computed rankings are written to

    params_dir = os.path.join("files", "params")

    return learn_data_path, results_path, figure_path, stats_file_path, log_path, path_rankings_test, path_rankings_val, params_dir, results_dir, rules_dir, path_rules, all_rule_types_false


def get_path_rankings_name(rankings_dir, rankings_name, dataset_name, split, options):
    """
    returns a specific path for the rankings file based on some of the parameters, e.g. confidence, aggregation function, ...
    :param rankings_dir: the directory path where the rankings file should be saved
    :param rankings_name: the name of the rankings file where the rankings should be written into
    :param dataset_name: the name of the dataset
    :param split: on which split do the evaluation take place, e.g. "test" for testset or "val" for validation set
    :param options: the options object
    """
    rankings_dataset_dir = os.path.join(rankings_dir, dataset_name)
    if not os.path.isdir(rankings_dataset_dir):
        os.makedirs(rankings_dataset_dir)
    
    name = "_"+split


    conf = options["THRESHOLD_CONFIDENCE"]
    if conf != 0.1:
        name += f"_conf_{str(conf)}"
    

    if not options["RULE_TYPE_CYC1_NON_REC"]:
        name += "_only_rec"

    if options["LEARN_PARAMS_OPTION"] == "default":
        name += "_default_config"
    
    corr_conf = options["THRESHOLD_CORRECT_PREDICTIONS"]
    if corr_conf != 10:
        name += f"_corr_conf_{str(corr_conf)}"

    agg_func = options["AGGREGATION_FUNCTION"] 
    name += f"_{agg_func}"

    if options["RULE_TYPE_C"]:
        name += "_crules"
    
    if options["RULE_TYPE_F"]:
        name += "_frules"

    if options["RULE_TYPE_Z"]:
        name += "_zrules"


    name+="_pvalue"
    name+=f"_{options['RULE_UNSEEN_NEGATIVES']}"
    
    num_top_rules = options["NUM_TOP_RULES"]
    if num_top_rules != -1:
        name += f"_num_top_rules_{str(num_top_rules)}"

    if options['MULTI_FLAG']:
        name += "_multi"

    
    return os.path.join(rankings_dataset_dir, rankings_name.format(name))

def get_path_rules_name(options, rules_dir, dataset_name):
    all_rule_types_false = True
    rule_name = '-'+options["LEARN_PARAMS_OPTION"]
    if options["RULE_TYPE_CYC1_NON_REC"]:
        rule_name+= '-cyc1nonrec'
        all_rule_types_false = False
    if options["RULE_TYPE_CYC1_REC"]:
        rule_name+= '-cyc1rec'
        all_rule_types_false = False
    if options["RULE_TYPE_C"]:
        rule_name+= '-crules'
        all_rule_types_false = False
    if options['MULTI_FLAG'] == True:
        rule_name += '-multi'
    rule_name+= '-'+ str(options["THRESHOLD_CORRECT_PREDICTIONS"]) + '-' + str(options["THRESHOLD_CONFIDENCE"]) 
    path_rules = os.path.join(rules_dir, dataset_name+rule_name+"-ruleset-{}.txt")     

    if all_rule_types_false:
        print("WARNING: All rule types are set to False. No rules will be learned or applied. You might run only on f and z rules.")
    return path_rules, all_rule_types_false

def write_ranksperrel(testmrrperrel, testhitsperrel, results_dir, dataset_name, testmode, z_factor=0):
    """ from the two given dicts, write the relation id (key) and the mrr/hits (value) to a txt file, 
    where each line is a relation id and the mrr/hits, and they are tab separated"""

    results_per_rel_dir = os.path.join(results_dir, "per_rel")
 
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if not os.path.isdir(results_per_rel_dir):
        os.makedirs(results_per_rel_dir)
    
    with open(os.path.join(results_per_rel_dir, f"mrrperrel_{testmode}_{dataset_name}_{z_factor}.txt"), "w") as file:
        file.write(f"rel\tmean_score\tnum_occurences\n")
        for key, value in testmrrperrel.items():
            file.write(f"{key}\t{value[0]}\t{value[1]}\n")
    
    with open(os.path.join(results_per_rel_dir, f"hits1perrel_{testmode}_{dataset_name}_{z_factor}.txt"), "w") as file:
        file.write(f"rel\tmean_score\tnum_occurences\n")
        for key, value in testhitsperrel.items():
            file.write(f"{key}\t{value[0]}\t{value[1]}\n")

def delete_rankings(path_rankings):
    """ delete the rankings file, if it exists"""
    if os.path.exists(path_rankings):
        os.remove(path_rankings)
        print(f"Deleted the file: {path_rankings}")
    else:
        print(f"The file {path_rankings} does not exist")  
        
def write_config_and_results(results_path, options, dataset_name, path_rankings_test, testmrr, testhits100, testhits10, testhits1, 
                             valmrr, valhits100, valhits10, valhits1, number_of_rules, mse_curvefit, large_data_flag, very_large_data_flag,
                             totaltime, evaltime, applytime, learntime, learndatatime):
    """ write the config and results to a csv file, that contains all params, results, runtime, etc.
     if the file already exists, append the results to the file."""



    
    
    params = {}
    params['datetime'] = get_date_time()
    params['config_dataset_name'] = dataset_name
    params['config_learn_option'] =  options["LEARN_PARAMS_OPTION"]
    params['config_multiflag'] = options["MULTI_FLAG"]    
    params['config_rankings_path'] = path_rankings_test
    params['config_eval_type'] = options["EVAL_TYPE"]
    params['config_large_data_flag'] = large_data_flag
    params['config_very_large_data_flag'] = very_large_data_flag
    params['config_rr_offset'] = options["RR_OFFSET"]
    params['config_aggregation_type'] = options["AGGREGATION_FUNCTION"]
    params['config_aggregation_decay'] = options["AGGREGATION_DECAY"]
    params['config_aggregation_numtoprules'] = options["NUM_TOP_RULES"]
    params['config_threshold_conf'] = options["THRESHOLD_CONFIDENCE"]
    params['config_threshold_corr_pred'] = options["THRESHOLD_CORRECT_PREDICTIONS"]
    params['config_pvalue'] = options["RULE_UNSEEN_NEGATIVES"]
    params['config_data_threshold_multi']= options["DATAPOINT_THRESHOLD_MULTI"]
    params['config_lmbda_reg'] = options["LMBDA_REG"]
    params['config_learn_window_size'] = options["LEARN_WINDOW_SIZE"]
    params['config_apply_window_size'] = options["APPLY_WINDOW_SIZE"]
    params['config_f_active'] = options["RULE_TYPE_F"]
    params['config_f_min_support'] = options["F_MIN_SUPPORT"]
    params['config_f_unseen_negatives'] = options["F_UNSEEN_NEGATIVES"]
    params['config_f_min_confidence'] = options["F_MIN_CONFIDENCE"]
    params['config_z_active'] = options["RULE_TYPE_Z"]
    if options["RULE_TYPE_Z"]:
        params['config_z_factor'] = options["Z_RULES_FACTOR"]
        params['config_z_min_support'] = options["Z_MIN_SUPPORT"]
        params['config_z_min_confidence'] = options["Z_MIN_CONFIDENCE"]
        params['config_z_unseen_negatives'] = options["Z_UNSEEN_NEGATIVES"]
    params['config_c_active'] = options["RULE_TYPE_C"]
    if options["RULE_TYPE_C"]: 
        params['config_c_conf'] = options["C_THRESHOLD_CONFIDENCE"]
        params['config_c_x_count'] = options["C_X_COUNT"]
        params['config_c_recurrency_active'] = options["C_RULE_RECURRENCY_ACTIVE"]
    if options["EVAL_TESTSET_FLAG"]:
        params['eval_testmrr'] = testmrr
        params['eval_testhits100'] = testhits100
        params['eval_testhits10'] = testhits10
        params['eval_testhits1'] = testhits1
    if options["EVAL_VALSET_FLAG"]:
        params['eval_valmrr'] = valmrr        
        params['eval_valhits100'] = valhits100
        params['eval_valhits10'] = valhits10
        params['eval_valhits1'] = valhits1
    params['time_totaltime'] = totaltime
    params['time_evaltime'] = evaltime
    params['time_applytime'] = applytime
    params['time_learntime'] = learntime
    params['time_learndatatime'] = learndatatime
    params['rule_reccuring_flag'] = options["RULE_TYPE_CYC1_REC"]
    params['rule_cyclic_flag'] = options["RULE_TYPE_CYC1_NON_REC"] 
    params['number_rules'] = number_of_rules
    params['fit_msecurvefit'] = mse_curvefit    



    write_results_csv(params, csv_filename=results_path)


def write_results_csv(params, csv_filename='results.csv'):
    # Check if the CSV file exists
    if os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0:
        # Load the existing CSV file
        df = pd.read_csv(csv_filename)
    else:
        # Create a new DataFrame with the column names from the params and vars
        df = pd.DataFrame()

    # Convert params to a DataFrame
    new_data = pd.DataFrame([params])

    # Combine the existing DataFrame with new_data
    # This ensures new columns are added and any missing data is filled with 'N/A'
    df = pd.concat([df, new_data], ignore_index=True).reindex(columns=df.columns.union(new_data.columns))
    # df = pd.concat([df, new_data], ignore_index=True).reindex(columns=(df.columns | new_data.columns))

    # Fill any missing values with 'N/A'
    df.fillna('N/A', inplace=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_filename, index=False)

    print(f"Data appended successfully to {csv_filename}.")


def get_date_time():

    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M")


def write_string_rankings(ranking_file_in_path, ruledataset, out_path=None):
    """ read the txt file in ranking_file_in_path and for every int that represents a node id, 
    write the corresponding string to a new file: either in out_path, or if out_path is None, in file
    with the same name but with the ending _string.txt"""
    if out_path is None:
        out_path = ranking_file_in_path.replace(".txt", "_string.txt")


    with open(ranking_file_in_path, "r") as file:
        with open(out_path, "w") as file2:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                head, relh, tail, t = lines[i].split(" ")
                head_string = ruledataset.nodes_id_to_string[int(head)]
                relh_string = ruledataset.rels_id_to_string[int(relh)]
                tail_string = ruledataset.nodes_id_to_string[int(tail)]

                file2.write(head_string[0] + "\t" + relh_string + "\t" + tail_string[0] + "\t" + str(t.split("\n")[0]) + "\n")

                if lines[i+1] != "\n":
                    candidates = lines[i+1].split(" ")
                    i = 0
                    for _ in range(0, len(candidates), 2):                       
                        if i < len(candidates)-2:
                            file2.write(ruledataset.nodes_id_to_string[int(candidates[i])][0] + "\t" + str(candidates[i+1]) + "\t")
                        else:
                            file2.write(ruledataset.nodes_id_to_string[int(candidates[i])][0] + "\t" + str(candidates[i+1]))

                        i+=2
  
                else:
                    file2.write("\n")


    print('wrote string rankings to: ', out_path)



    
