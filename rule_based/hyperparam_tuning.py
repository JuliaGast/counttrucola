from main import main
import argparse
from options import Options

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="tkgl-icews0515", help="dataset name to use for hyperparameter tuning")
args, unknown_args = parser.parse_known_args()    
dataset_name = args.dataset  

# depending on the dataset, we set the dataset size -> thepending on the size, we test more or less hyperparameters
if dataset_name in ["tkgl-icews14", "tkgl-wikiold", "tkgl-yago", "tkgl-smallpedia", 'tkgl-monkey']:
    dataset_size = 'small'
elif dataset_name in ['tkgl-icews18', 'tkgl-icews0515']:
    dataset_size = 'medium' 
elif dataset_name == 'icews':
    dataset_size = 'very_large'
else:
    dataset_size = 'large'

if dataset_size == 'small':
    ## GROUP1 : params for learning rules
    # 8*3 + 3  = 27
    RULE_UNSEEN_NEGATIVES = [0,1,2,3,5,10,20,30]
    DATAPOINT_THRESHOLD_MULTI = [0, 10, 50]
    LEARN_WINDOW_SIZE = [50,100,150] # Independent
    if dataset_name == "tkgl-icews14":
        LEARN_WINDOW_SIZE = [50, 100, 150] 
    elif dataset_name == "tkgl-wikiold":
        LEARN_WINDOW_SIZE = [5, 10, 30, 50, 100]
    elif dataset_name == "tkgl-yago" or dataset_name == "tkgl-smallpedia":
        LEARN_WINDOW_SIZE = [5, 10, 30, 30, 50]
    elif dataset_name == "tkgl-monkey":
        LEARN_WINDOW_SIZE = [10, 12, 15]
    else:
        LEARN_WINDOW_SIZE = [50,100,150] # Independent
    # GROUP 2: f and z rule params
    # 7*6 = 42
    # no learning
    Z_RULES_FACTOR= [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0] # Z rules factor
    F_UNSEEN_NEGATIVES = [0, 1, 5, 10, 20, 30] # F unseen negatives
    # Group 3: C rules and RR_Offset
    #  3* 2  = 6
    RR_OFFSET = [-99, 0, -1]  # learn data creation redundancy;
    RULE_TYPE_C = [True, False]  # C rules
    # Group 4: aggregation functions
    #3* 5 = 15
    # no learning
    aggregation_functions = ['noisyor']
    num_top_rules = [5, 10, 50] 
    AGGREGATION_DECAY = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
elif dataset_size == 'medium':
    ## GROUP1 : params for learning rules
    # 3 + 3  = 6
    RULE_UNSEEN_NEGATIVES = [1,5,10]
    DATAPOINT_THRESHOLD_MULTI = [50]
    if dataset_name =='tkgl-icews0515':
        LEARN_WINDOW_SIZE = [50, 100, 150, 200, 250] # Independent
    else:
        LEARN_WINDOW_SIZE = [50,100,150] # Independent
    # GROUP 2: f and z rule params
    # 3*3 = 9
    # no learning
    Z_RULES_FACTOR= [0, 0.1,0.5] # Z rules factor
    F_UNSEEN_NEGATIVES = [0,  10, 30] # F unseen negatives
    # Group 3: C rules and RR_Offset
    #  3* 2  = 6
    RR_OFFSET = [-99, 0, -1]  # learn data creation redundancy;
    RULE_TYPE_C = [True, False]  # C rules
    # Group 4: aggregation functions
    #2* 3 = 6
    # no learning
    aggregation_functions = ['noisyor']
    num_top_rules = [10, 50] 
    AGGREGATION_DECAY = [1, 0.8, 0.6]
elif dataset_size == 'large':
    ## GROUP1 : params for learning rules
    # 2 + 3  = 5
    RULE_UNSEEN_NEGATIVES = [1,10]
    DATAPOINT_THRESHOLD_MULTI = [50]
    LEARN_WINDOW_SIZE = [100,150,200] # Independent
    # GROUP 2: f and z rule params
    # 3*3 = 9
    # no learning
    Z_RULES_FACTOR= [0, 0.1, 0.5] # Z rules factor
    F_UNSEEN_NEGATIVES = [10] # F unseen negatives
    # Group 3: C rules and RR_Offset
    #  2* 2  = 4
    RR_OFFSET = [-99, -1]  # learn data creation redundancy;
    RULE_TYPE_C = [True, False]  # C rules
    # Group 4: aggregation functions
    # 1
    # no learning
    aggregation_functions = ['noisyor']
    num_top_rules = [10] 
    AGGREGATION_DECAY = [0.8]
elif dataset_size == 'very_large':
    ## GROUP1 : params for learning rules
    # 2 + 3  = 5
    RULE_UNSEEN_NEGATIVES = [1,10]
    DATAPOINT_THRESHOLD_MULTI = [50]
    LEARN_WINDOW_SIZE = [100,150,200] # Independent
    # GROUP 2: f and z rule params
    # 3*3 = 9
    # no learning
    Z_RULES_FACTOR= [0, 0.1, 0.5] # Z rules factor
    F_UNSEEN_NEGATIVES = [10] # F unseen negatives
    # Group 3: C rules and RR_Offset
    #  2* 2  = 4
    RR_OFFSET = [-99, -1]  # learn data creation redundancy;
    RULE_TYPE_C = [False]  # C rules
    # Group 4: aggregation functions
    # 1
    # no learning
    aggregation_functions = ['noisyor']
    num_top_rules = [10] 
    AGGREGATION_DECAY = [0.8]




options_call = {}
options_call["DATASET_NAME"] = dataset_name

options_call["RULE_TYPE_F"] = True
options_call["RULE_TYPE_Z"] = True

options_obj = Options(config_file_name="config-default.yaml")
options = options_obj.options
best_config = {}
best_val_mrr = 0
# these are the default values to start with. potentially they will be overwritten in the following.
# group 1

best_config["RULE_UNSEEN_NEGATIVES"] = options["RULE_UNSEEN_NEGATIVES"]
best_config["DATAPOINT_THRESHOLD_MULTI"] = options["DATAPOINT_THRESHOLD_MULTI"]
if not dataset_name == 'tkgl-monkey':
    best_config["LEARN_WINDOW_SIZE"] = options["LEARN_WINDOW_SIZE"]
else:
    best_config["LEARN_WINDOW_SIZE"] = 10
# group 2
best_config["Z_RULES_FACTOR"] = options["Z_RULES_FACTOR"]
best_config["F_UNSEEN_NEGATIVES"] = options["F_UNSEEN_NEGATIVES"]
# group 3
best_config["RULE_TYPE_C"] = True
best_config["RR_OFFSET"] = options["RR_OFFSET"]
# group 4
best_config["NUM_TOP_RULES"] = options["NUM_TOP_RULES"]
best_config["AGGREGATION_DECAY"] = options["AGGREGATION_DECAY"]


options_call.update(best_config)

## Group 1
print("----------------------Starting hyperparameter tuning for group 1----------------------")

create_learn_data_flag = True
for unseen_negatives in RULE_UNSEEN_NEGATIVES:
    for datapoint_threshold_multi in DATAPOINT_THRESHOLD_MULTI:
        options_call["DATAPOINT_THRESHOLD_MULTI"] = datapoint_threshold_multi
        options_call["RULE_UNSEEN_NEGATIVES"] = unseen_negatives
        options_call["CREATE_LEARN_DATA_FLAG"] = create_learn_data_flag
        val_mrr = main(options_call=options_call)
        if ((val_mrr - best_val_mrr) > 0.0001):
            best_val_mrr = val_mrr
            best_config["RULE_UNSEEN_NEGATIVES"] = unseen_negatives
            best_config["DATAPOINT_THRESHOLD_MULTI"] = datapoint_threshold_multi
        create_learn_data_flag = False  # we do not need to create the learn data again

print("Best configuration after group 1a):"
      f"{best_config}")

options_call.update(best_config)
for window_size in LEARN_WINDOW_SIZE:  
    options_call["LEARN_WINDOW_SIZE"] = window_size
    if window_size != options["LEARN_WINDOW_SIZE"]:
        options_call["CREATE_LEARN_DATA_FLAG"] = True  # we do not need to create the learn data again
    else:
        options_call["CREATE_LEARN_DATA_FLAG"] = False
    options_call["LOAD_PARAMS_FLAG"] = False

    val_mrr = main(options_call=options_call)
    if ((val_mrr - best_val_mrr) > 0.0001):
        best_val_mrr = val_mrr
        best_config["LEARN_WINDOW_SIZE"] = window_size

print("Best configuration after group 1:"
      f"{best_config}")

## Group 2
print("----------------------Starting hyperparameter tuning for group 2----------------------")
options_call.update(best_config)
options_call["CREATE_LEARN_DATA_FLAG"] = False # we have learn data for each window size already
options_call["LOAD_PARAMS_FLAG"] = False  # learn rules once more for best config

for z_rules_factor in Z_RULES_FACTOR:
    options_call["Z_RULES_FACTOR"] = z_rules_factor
    for f_unseen_negatives in F_UNSEEN_NEGATIVES:
        options_call["F_UNSEEN_NEGATIVES"] = f_unseen_negatives
        val_mrr = main(options_call=options_call)
        if ((val_mrr - best_val_mrr) > 0.0001):
            best_val_mrr = val_mrr
            best_config["Z_RULES_FACTOR"] = z_rules_factor
            best_config["F_UNSEEN_NEGATIVES"] = f_unseen_negatives
        options_call["LOAD_PARAMS_FLAG"]  = True

print("Best configuration after group 2:"
      f"{best_config}")

## Group 3 
print("----------------------Starting hyperparameter tuning for group 3----------------------")
options_call.update(best_config)
options_call["LOAD_PARAMS_FLAG"]  = True
options_call["CREATE_LEARN_DATA_FLAG"] = False  # we do not need to create the learn data again

for rule_type_c in RULE_TYPE_C:
    options_call["RULE_TYPE_C"] = rule_type_c
    for RR_OFFSET_value in RR_OFFSET:
        options_call["RR_OFFSET"] = RR_OFFSET_value
        if RR_OFFSET_value == options["RR_OFFSET"]: 
            options_call["CREATE_LEARN_DATA_FLAG"] = False # learn data has alreadybeen created with this RR_OFFSET
        elif rule_type_c == False:
            options_call["CREATE_LEARN_DATA_FLAG"] = False # learn data has already been created with c rules
        else:
            options_call["CREATE_LEARN_DATA_FLAG"] = True
        options_call["LOAD_PARAMS_FLAG"]  = False
        val_mrr = main(options_call=options_call)
        if ((val_mrr - best_val_mrr) > 0.0001):
            best_val_mrr = val_mrr
            best_config["RULE_TYPE_C"] = rule_type_c
            best_config["RR_OFFSET"] = RR_OFFSET_value

print("Best configuration after group 3:"
      f"{best_config}")
## Group 4
print("----------------------Starting hyperparameter tuning for group 4----------------------")
options_call.update(best_config)
options_call["CREATE_LEARN_DATA_FLAG"] = False
options_call["LOAD_PARAMS_FLAG"] = False

for num_top_rule in num_top_rules:
    options_call["NUM_TOP_RULES"] = num_top_rule            
    for decay in AGGREGATION_DECAY:
        options_call["AGGREGATION_DECAY"] = decay
        val_mrr = main(options_call=options_call)
        if ((val_mrr - best_val_mrr) > 0.0001):
            best_val_mrr = val_mrr
            best_config["NUM_TOP_RULES"] = num_top_rule
            best_config["AGGREGATION_DECAY"] = decay
        options_call["CREATE_LEARN_DATA_FLAG"] = False
        options_call["LOAD_PARAMS_FLAG"] = True

print("Best configuration after group 4:"
      f"{best_config}")
## done

print("Best configuration found:")
for key, value in best_config.items():
    print(f"{key}: {value}")
print(f"Best validation MRR: {best_val_mrr}")



print("Running final evaluation on test set with best configuration...")
options_call.update(best_config)
options_call["EVAL_TESTSET_FLAG"] = True
options_call["CREATE_LEARN_DATA_FLAG"] = True
options_call["LOAD_PARAMS_FLAG"] = False  
val_mrr = main(options_call=options_call)