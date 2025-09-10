import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../rule_based")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rule_based.rule_dataset import RuleDataset
import rule_based.rule_utils as rule_utils
import rule_based.utils as utils
from tgb.linkproppred.evaluate import Evaluator


# we want to compare the rankings in two different files. rankings a and rankings b.
# we want to find the quadruples where the ranking in b is [significantly] better than in a
# [significantly], means that we have the following conditions:
# 1. the correct candidate in b has a higher rank than in a
# 2. the correct candidate in b has a rank 1
# 3. no other candidate in b has the same score as the ground truth candidate
# 4. backlog + lead > diff_threshold, where 
#       backlog is the difference between the score of the correct candidate in a and the highest score of the negative (wrong, but after time-filter) examples in a
#       lead is the difference between the score of the correct candidate in b and the highest score of the negative (wrong, but after time-filter) examples in b
#       diff_threshold is a user-set threshold 


def compare_rankings(rankings_worse_name, rankings_better_name, path, evaluation_mode, filebetter_path, explanations_path, dataset_name, relations_of_interest=[], diff_threshold=0.0005):
    """ compare the rankings of two different ranking files and write the quadruples where the rankings in rankings_better_name are significantly 
    better than in rankings_worse_name to a file. you can compare the rankings either for all relations or only for a subset of relations (relations_of_interest).
    the output file can be used as input (quadruples.txt) for the explainer and is automatically put into the directory explanations_path.
    Further, create a file that contains the quadruples where the rankings in rankings_better_name are significantly better than in rankings_worse_name and the ranks
    of both files for the quadruples to filebetter_path.
    significantly better means that the following conditions are met:
    1. the correct candidate in rankings_better_name has a higher rank than in rankings_worse_name
    2. the correct candidate in rankings_better_name has a rank 1
    3. no other candidate in rankings_better_name has the same score as the ground truth candidate
    4. backlog + lead > diff_threshold, where backlog is the difference between the score of the correct candidate in rankings_better_name and the highest score 
    of the negative examples in rankings_better_name, lead is the difference between the score of the correct candidate in rankings_worse_name and the highest 
    score of the negative examples in rankings_worse_name, and diff_threshold is a user-set threshold
    :param rankings_worse_name: name of the file that contains the rankings that are worse
    :param rankings_better_name: name of the file that contains the rankings that are better
    :param path: path to the directory that contains the ranking files
    :param evaluation_mode: 'val' or 'test', depending on which set you want to evaluate
    :param filebetter_path: path to the output file that contains the quadruples where the rankings in rankings_better_name are significantly better than in rankings_worse_name
    :param explanations_path: path to the output file that contains the quadruples where the rankings in rankings_better_name are significantly better than in rankings_worse_name, 
    this can be used as input for the explainer
    :param relations_of_interest: list of relations for which the rankings should be compared, leave empty to compare for all relations
    :param diff_threshold: threshold for backlog + lead to consider the better ranking as significantly better
    :return: a dataframe that contains the quadruples where the rankings in rankings_better_name are significantly better than in rankings_worse_name
    and the ranks of both files for the quadruples
    """

    rule_dataset = RuleDataset(name=dataset_name)
    dataset = rule_dataset.dataset
    num_nodes = rule_dataset.dataset.num_nodes
    split_mode = evaluation_mode
    evaluator = Evaluator(name=dataset.name, k_value=[1,10,100])
    neg_sampler = dataset.negative_sampler  

    if evaluation_mode == "val":
        testdata = rule_dataset.val_data
        print("loading negative val samples")
        dataset.load_val_ns() # load negative samples, i.e. the nodes that are not used for time aware filter mrr
    elif evaluation_mode == "test":
        testdata = rule_dataset.test_data
        print("loading negative test samples")
        dataset.load_test_ns() # load negative samples, i.e. the nodes that are not used for time aware filter mrr

    rankings_worse_rules = rule_utils.read_rankings_order(os.path.join(path,rankings_worse_name), num_nodes)
    rankings_better_rules = rule_utils.read_rankings_order(os.path.join(path,rankings_better_name), num_nodes)

    b_better ={}

    print('>>> starting evaluation for every triple, in the ', evaluation_mode, 'set')
    total_iterations = len(testdata)
    progressbar_percentage = 0.01

    i_list = []
    increment = int(total_iterations*progressbar_percentage) if int(total_iterations*progressbar_percentage) >=1 else 1
    remaining = total_iterations
    mrr_per_rel = {}
    hit1_per_rel = {}


    mrr_per_rel_brules = {}
    hit1_per_rel_brules = {}
    with open(filebetter_path, 'w') as filebetter:
        with open(explanations_path, 'w') as file_explanations:
            with tqdm(total=total_iterations) as pbar:
                counter = 0
                filebetter.write("src\trel\tdst\tt\trank_worse_file\trank_better_file\n") #header
                file_explanations.write("subject rel object timestep\n")
                for i, (src, dst, t, rel) in enumerate(zip(testdata[:,0], testdata[:,2], testdata[:,3], testdata[:,1])):
                    # Update progress bar
                    if len(relations_of_interest) > 0 and rel not in relations_of_interest: # only compare for relations of interest
                        continue
                    counter += 1
                    if counter % increment == 0:
                        remaining -= increment
                        pbar.update(increment)
                    if remaining < increment:
                        pbar.update(remaining)
                        
                    original_t = rule_dataset.timestamp_id2orig[t]

                    # Query negative batch list - all negative samples for the given positive edge that are not temporal conflicts (time aware mrr)
                    neg_batch_list = neg_sampler.query_batch(np.array([src]), np.array([dst]), np.array([original_t]), edge_type=np.array([rel]), split_mode=split_mode)

                    # Make predictions for given src, rel, t
                    # Compute a score for each node in neg_batch_list and for actual correct node dst
                    scores_array_worse_rules =rule_utils.create_scores_array(rankings_worse_rules[(src, rel, t)], num_nodes)
                    scores_array_better_rules =rule_utils.create_scores_array(rankings_better_rules[(src, rel, t)], num_nodes)

                    #### worse_rules
                    predictions_neg_worse_rules = scores_array_worse_rules[neg_batch_list[0]]
                    predictions_pos_worse_rules = np.array(scores_array_worse_rules[dst])
                    # Evaluate the predictions
                    input_dict = {
                        "y_pred_pos": predictions_pos_worse_rules,
                        "y_pred_neg": predictions_neg_worse_rules,
                        "eval_metric": ['mrr'], 
                    }
                    predictions_worse_rules = evaluator.eval(input_dict)
                    predictions_worse_rules['rank'] = int(1/predictions_worse_rules['mrr'])

                    i_list.append(i)
                    if rel not in mrr_per_rel:
                        mrr_per_rel[rel] = (float(predictions_worse_rules['mrr']), 1)
                        hit1_per_rel[rel] = (float(predictions_worse_rules['hits@1']), 1)
                    else:
                        mrr_per_rel[rel] = (mrr_per_rel[rel][0] + float(predictions_worse_rules['mrr']), mrr_per_rel[rel][1] + 1)
                        hit1_per_rel[rel] = (hit1_per_rel[rel][0] + float(predictions_worse_rules['hits@1']), hit1_per_rel[rel][1] + 1)

                    #### better_rules
                    predictions_neg_better_rules = scores_array_better_rules[neg_batch_list[0]]
                    predictions_pos_better_rules = np.array(scores_array_better_rules[dst])
                    # Evaluate the predictions
                    input_dict = {
                        "y_pred_pos": predictions_pos_better_rules,
                        "y_pred_neg": predictions_neg_better_rules,
                        "eval_metric": ['mrr'],
                    }
                    predictions_better_rules = evaluator.eval(input_dict)
                    predictions_better_rules['rank'] = int(1/predictions_better_rules['mrr'])

                    i_list.append(i)
                    if rel not in mrr_per_rel_brules:
                        mrr_per_rel_brules[rel] = (float(predictions_better_rules['mrr']), 1)
                        hit1_per_rel_brules[rel] = (float(predictions_better_rules['hits@1']), 1)
                    else:
                        mrr_per_rel_brules[rel] = (mrr_per_rel_brules[rel][0] + float(predictions_better_rules['mrr']), mrr_per_rel_brules[rel][1] + 1)
                        hit1_per_rel_brules[rel] = (hit1_per_rel_brules[rel][0] + float(predictions_better_rules['hits@1']), hit1_per_rel_brules[rel][1] + 1)

                    #### check if all conditions are met to have b-predictiion [significantly] better than a-prediction
                    if float(predictions_worse_rules['mrr']) < float(predictions_better_rules['mrr']): #1) the correct candidate in b has a higher rank than in a
                        if float(predictions_better_rules['mrr']) == float(1): # > float(0.05): #2) the correct candidate in b has a rank 1
                            # get predictions for negative examples which are == positive prediction
                            predictions_neg_brules_equal = predictions_neg_better_rules[predictions_neg_better_rules == predictions_pos_better_rules]
                            if len (predictions_neg_brules_equal) == 0: #3) no other candidate in b has the same score as the ground truth candidate
                                # get predictions for negative examples which are < positive prediction
                                predictions_neg_brules_lower = predictions_neg_better_rules[predictions_neg_better_rules < predictions_pos_better_rules]
                                lead = predictions_pos_better_rules - max(predictions_neg_brules_lower) # lead is the difference between the score of the correct candidate in b and the highest score of the negative (wrong, but after time-filter) examples in b
                                # backlog von richtigem kandidaten auf den hoechsten predicteten score:
                                backlog = max(predictions_neg_worse_rules) - predictions_pos_worse_rules # backlog is the difference between the score of the correct candidate in a and the highest score of the negative (wrong, but after time-filter) examples in a
                                if (lead + backlog) > diff_threshold: #4) backlog + lead > diff_threshold
                                    b_better[(src, rel, dst, t)] = (float(predictions_worse_rules['mrr']), float(predictions_better_rules['mrr']), predictions_pos_worse_rules, predictions_pos_better_rules)
                                    # write src rel dst t to txt file
                                    filebetter.write(str(src) + "\t" + str(rel) + "\t" + str(dst) + "\t" + str(t) +
                                        "\t"   +  str(int(predictions_worse_rules['rank'])) + "\t" + str(int(predictions_better_rules['rank'])) + '\n')
                                    # Add to DataFrame list
                                    if 'df_rows' not in locals():
                                        df_rows = []
                                    df_rows.append({
                                        'src': src,
                                        'rel': rel,
                                        'dst': dst,
                                        't': t,
                                        'rank_worse': int(predictions_worse_rules['rank']),
                                        'rank_better': int(predictions_better_rules['rank'])
                                    })
                                    # write src rel dst t to txt file
                                    file_explanations.write(str(src) + " " + str(rel) + " " + str(dst) + " " + str(t) + '\n')
                

    df_compare = pd.DataFrame(df_rows)

    print("in total we had", len(b_better), "cases where the rankings in file better_rule performed [significantly] better than the rankings in file worse_rule and the better_rule had hits@1 of 1")
    print("the quadruples are written to", explanations_path, "and can be used as input for the explainer")
    print("the quadruples and the ranks in both files are written to", filebetter_path)
    return df_compare


if __name__ == "__main__":

    diff_threshold = 0.0005 # threshold for backlog + lead to consider the better ranking as significantly better    

    dataset_name = 'tkgl-icews14'
    experiment_name = 'regcn4' # name for the explainer experiment, where the output will be stored
    # path to the worse rankings:
    evaluation_mode = 'test'
    rankings_worse_name = 'ICEWS14_rankings_regcn.txt'  
    relations_of_interest = [4] # leave empty for all relations, otherwise only compare for the given relations

    # path to the better rankings:
    rankings_better_name = 'tkgl-icews14-rankings_test_conf_0_corr_conf_0_noisyor_crules_frules_zrules_pvalue_30_num_top_rules_10_multi.txt' 

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'files', 'rankings', dataset_name)

    # path to output - this can be used as input for the explainer
    outpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'files', 'explanations', experiment_name, 'input')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filebetter_path = os.path.join(outpath, "filesb_better_all_"+dataset_name+".txt")
    explanations_path = os.path.join(outpath, "quadruples.txt")  # this can be used as input for the explainer

    compare_df = compare_rankings(rankings_worse_name, rankings_better_name, path, evaluation_mode, filebetter_path, explanations_path, rule_dataset=None, relations_of_interest=relations_of_interest, diff_threshold=diff_threshold)