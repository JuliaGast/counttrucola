"""
please put the rules of interest in the folder files/explanations/1/input
Needed:
txt file with rules rulefile, with naming ideally: dataset name (e.g. tkgl-icews) + - + ... + -ruleset-ids.txt, e.g.:  tkgl-icews14-whateveryouwant-ruleset-ids.txt, and ruleset-strings.txt
txt file with quadruples quadruples.txt, in format subject_id, rel_id, object_id, timestep - if not specified you can alternatively speciy your quadruple as user input in terminal

"""
import os
import explainer_utils
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../rule_based")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rule_based.eval as eval
import rule_based.utils as utils

from IPython.display import display, HTML
import webbrowser

## explainer config
exp_name = '1'
dataset_split = 'test' # which part of the dataset are the quadruples from? 'test' or 'val'

plot_figures_flag=True
recreate_figures = False #True # do you want to recreate the figures? If yes, all old figures will be deleted, if no, the old figures for exp_name will be reused

explain_all_quads_flag = False # explain all quads in the val or test set, instead of quads from quadruples.txt. this might be slow if you have a large dataset
max_rules_per_pred = 30 # how many rules shoul be shown per prediction? If there are more rules, the top ones will be shown based on their score.
num_cpus= 1 # number of cpus to use for parallel processing 

## CountTRuCoLa config
## If you do not know what to do, leave them None, then the default params will be used.
AGGREGATION_FUNCTION, NUM_TOP_RULES, AGGREGATION_DECAY, F_UNSEEN_NEGATIVES, Z_RULES_FACTOR, APPLY_WINDOW_SIZE, RULE_TYPE_Z_FLAG, RULE_TYPE_F_FLAG = None, None, None, None, None, None, None, None

## Otherwise, change the values here: (uncomment if you want)
# ## a) rule aggregation
# AGGREGATION_FUNCTION = "noisyor" # select from "maxplus" / "noisyor" / "max" / 
# NUM_TOP_RULES = 5 #  noisy-or top-h; stops adding predicting rules to a candidate of a query if already num_top_rules; -1 means no limit
# # predicted the candidate; if all candidates are predicted by num_top_rules, rule
# # application is stopped; can be used in conjunction with "noisyor" to achieve
# AGGREGATION_DECAY = 0.8  # decay factor for the aggregation function; only used for "noisyor"; the second score is multiplied by decay, the third by decay^2 and so on; if set to 1, no decay is applied; 

# ## b) f and z-rules
# RULE_TYPE_Z_FLAG = True  # do you want to use z-rules
# RULE_TYPE_F_FLAG = True # do you want to use f-rules 
# F_UNSEEN_NEGATIVES = 30  # A constant added to the denominator when computing confidences for f-rules. 
# Z_RULES_FACTOR = 0.1 # A scaling factor Z ∈ [0, 1] applied to the score predicted by the z-rules.

# # c) window size for rule applciation
# APPLY_WINDOW_SIZE = -1 # how many previous interactions do we take into account for the rules (for apply) - recommend: set to -1, to use all timesteps; or set as large as possible

## Explainer steps
# prepare paths
in_folder, out_folder = explainer_utils.prepare_paths(exp_name, recreate_figures)

# get the data from user input: what quadruples to explain? what dataset to use?
dataset, dataset_name, testset_dict, path_rules, quadruples = explainer_utils.get_data_from_user_input(in_folder, dataset_split, explain_all_quads_flag)

# set all options
user_options = {
    "AGGREGATION_FUNCTION": AGGREGATION_FUNCTION,
    "NUM_TOP_RULES": NUM_TOP_RULES,
    "AGGREGATION_DECAY": AGGREGATION_DECAY,
    "F_UNSEEN_NEGATIVES": F_UNSEEN_NEGATIVES,
    "Z_RULES_FACTOR": Z_RULES_FACTOR,
    "APPLY_WINDOW_SIZE": APPLY_WINDOW_SIZE,
    "RULE_TYPE_Z": RULE_TYPE_Z_FLAG,
    "RULE_TYPE_F": RULE_TYPE_F_FLAG
}

options_explain = explainer_utils.set_options( user_options, num_cpus, dataset_name, config_path = "")
print("Options for the explainer:")
print(options_explain)

# explain
num_rules, rule_triple_dict = explainer_utils.explain(dataset,out_folder, dataset_split, path_rules=path_rules, options_explain=options_explain,
                         max_rules_per_pred=max_rules_per_pred, plot_figures_flag=plot_figures_flag)

print("You can find the explanations here:")
print(out_folder)
html_file = os.path.join(out_folder, "explanations_fancy.html")

# Display a clickable link 
webbrowser.open_new_tab(html_file)


# eval
path_rankings = os.path.join(out_folder,'ranks.txt')
mrr, hits10, hits1, hits100, mrrperrel,hits1perrel = eval.evaluate(dataset, path_rankings, 0.01, evaluation_mode=dataset_split, eval_type='random', special_evalquads=quadruples)
utils.write_ranksperrel(mrrperrel, hits1perrel, out_folder, dataset.dataset.name, 'val')

print('mrr: ', mrr)
print(f'hits@1, hits@10, hits@100: {hits1}, {hits10}, {hits100}')

## optional: make some statistics
# explainer_utils.make_rule_stats()
# we could also do stats like:
# the rule with the most predictions in this dataset was: xxx
# the rule with the highest average prediction was: xxx
# the most successful rule (i.e. highest mrr oder so) was: xxx
# "if we would remove the recurrency rules we would get.."