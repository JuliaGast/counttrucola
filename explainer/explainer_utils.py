import os
import sys
from tabnanny import verbose
import numpy as np
from matplotlib import pyplot as plt
import json
import matplotlib.colors as mcolors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../rule_based")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rule_based.rule_dataset import RuleDataset
from rule_based.rules import RuleSet
import rule_based.predictor as predictor
from rule_based.rules import Rule1, Rule2, RuleC, RuleCBackward
from rule_based.options import Options

from rule_based.rule_utils import score_single, score_linear


def set_options(user_options, num_cpus, dataset_name, config_path=None):
    """ get the default options from config file.
    for each configuration that has been specified by the user in user_options (each entry that is not None): overwrite options.
    param user_options: dict, keys: config key, value config value
    num_cpus: int, number of CPUs to use
    config_path: string, path to the config directory yaml file

    """

    # todo overwrite dataset name in options with 'dataset_name;
    options_call = {}
    options_call["DATASET_NAME"] = dataset_name
    if config_path:
        options = Options(config_file_name=os.path.join(config_path, "config-default.yaml"), options_call=options_call)
    else:
        options = Options(config_file_name="config-default.yaml", options_call=options_call)
    options.parse_options()
    options = options.options

    options_explain = options
    for key, value in user_options.items():
        if value is not None:
            options_explain[key] = value
    options_explain['NUM_CPUS'] = num_cpus

    return options_explain


def prepare_paths(exp_name, recreate_figures, jupyter_flag=False):    
    """ Prepare input and output paths for the explainer. and copy style.css file to respective folders
    :param exp_name: string, name of your experiment folder, where you have the rules and quadruples.txt
    :param recreate_figures: boolean, whether to recreate figures; if True: all figures in out folder will be deleted; if False: they will be reused if possible
    :jupyter_flag: boolean, whether the code is running in a Jupyter notebook (this affects the input/output paths)
    :returns:
        in_folder: string, path to the input folder
        out_folder: string, path to the output folder
    """
    if jupyter_flag:
        in_folder = os.path.join("..", "files", "explanations", exp_name, "input")
        out_folder = os.path.join("..", "files", "explanations", exp_name, "output")
        style_src = os.path.join("..", "files", "explanations", "styles.css")
    else:
        in_folder = os.path.join("files", "explanations", exp_name, "input")
        out_folder = os.path.join("files", "explanations", exp_name, "output")
        style_src = os.path.join("files", "explanations", "styles.css")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Copy style.css from ./files/explanations to out_folder
    style_dst = os.path.join(out_folder, "styles.css")
    if os.path.exists(style_src):
        if os.path.exists(style_dst):
            os.remove(style_dst)
        with open(style_src, 'r') as src_file:
            with open(style_dst, 'w') as dst_file:
                dst_file.write(src_file.read())
        print(f"Successfully copied {style_src} to {style_dst}.")
    else:
        print(f"Warning: Could not find {style_src}. The output will not be styled.")


    if recreate_figures: # delete the folder and all figures inside, they will be created new
        print('you decided to recreate the figures. Thus I will delete the old figures which are stored in: ', os.path.join(out_folder, "figures"))
        figures_folder = os.path.join(out_folder, "figures")
        if os.path.exists(figures_folder): 
            for filename in os.listdir(figures_folder):
                file_path = os.path.join(figures_folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            os.rmdir(figures_folder)
    else:
        print('you decided to reuse the old figures which are stored in: ', os.path.join(out_folder, "figures"))

    return in_folder, out_folder



def get_data_from_user_input(in_folder, dataset_split, explain_all_quads_flag):
    """ get data that is needed for explanation based on user input
    check if the needed files are in the folder. if quadruples.txt is not there, ask for user input
    check if the quadruples are in the test set. if not, throw error
    :param in_folder: [str] path to the folder containing the ruleset file and the quadruples file
    :param dataset_split: [str] which part of the dataset are the quadruples from? 'test' or 'valid'
    :param explain_all_quads_flag: [bool] if true: explain all quads in the val or test set, instead of quads from quadruples.txt

    :return: dataset: [RuleDataset] dataset containing the nodes, relations, and so on
    :return: dataset_name: [str] name of the dataset, e.g. 'tkgl-icews14'
    :return: testset: [dict] test or valid (see dataset_split) set containing the quadruples in form [subject_id][rel_id][object_id][timestep]
    :return: rule_path: path to the rule file
    :return: quadruples: [list] list of quadruples [subject_id, rel_id, object_id, timestep]
    
    """

    ## look for ruleset file 
    ruleset_file = None
    for file_name in os.listdir(in_folder):
        if "-id" in file_name:
            ruleset_file = file_name
            break
    if ruleset_file:
        print("I will use the rules specified in the following file:")    
        print(os.path.join(in_folder, ruleset_file))
        dataset_name = "-".join(ruleset_file.split("-")[0:2])
        if not 'tkgl' in dataset_name:
            dataset_name = input("Please provide the dataaset name in the format tkgl-datasetname, e.g. tkgl-icews14: ")
        print("Operating on dataset: ", dataset_name)
    else:
        print("No file containing '-id' found in the folder. Ending this explanation. Please name your ruleset file accordingly and restart.")
        print(f"Please put a file named 'datasetname-whateveryouwant-id.txt' (e.g. tkgl-icews14-learn-cyc1nonrec-cyc1rec-crules-0-0-0-ruleset-ids.txt) in the folder. {in_folder}")
        sys.exit(0)

    ## create dataset object and load dataset
    dataset = RuleDataset(name=dataset_name)

    ## read rules from file
    print(f'I will use rules and params from file {ruleset_file}')
    rule_path = os.path.join(in_folder, ruleset_file)
    testset = dataset.head_rel_tail_t[dataset_split]

    ## find the quadruples that should be explained
    if explain_all_quads_flag: # explain all
        print(f"I will explain all quadruples in the [{dataset_split}] set.")
        print("if you have a dataset which is not tiny (>30 quads), this might take a while.")
        if dataset_split== 'test':
            quadruples = dataset.test_data
        else:
            quadruples = dataset.val_data
    else: # look for quadruples file or ask for quadruples input. predictions for these quadruples shoule be explained
        nodes_string_to_id = {v[0]: k for k, v in dataset.nodes_id_to_string.items()}
        rels_string_to_id = {v: k for k, v in dataset.rels_id_to_string.items()}
        
        quadruples_file = os.path.join(in_folder, "quadruples.txt")

        if os.path.exists(quadruples_file):
            print("I will explain the quadruples specified in the following file:")
            print(quadruples_file)
            if dataset_split== 'test':
                valtest_data = dataset.test_data
            else:
                valtest_data = dataset.val_data
            quadruples = read_quads(quadruples_file, valtest_data,nodes_string_to_id, rels_string_to_id)

            print(f"I will explain in total {len(quadruples)} quadruples.")
            if len(quadruples) <6:
                print("I will explain the following quadruples:")
                for quad in quadruples:
                    print(quad)
                    try:
                        sub_string = dataset.nodes_id_to_string[quad[0]]
                        rel_string = dataset.rels_id_to_string[quad[1]]
                        obj_string = dataset.nodes_id_to_string[quad[2]]            
                    except KeyError as e:
                        print("Please provide a valid quadruple.")
                        print(e, "not in the dataset.")
                    print(f"{sub_string} {rel_string} {obj_string} {quad[3]}")
        else: #user input
            print("No file named 'quadruples.txt' found in the folder.")
            valtest_data = dataset.test_data if dataset_split == 'test' else dataset.val_data
            timesteps_min = valtest_data[:, 3].min()
            timesteps_max = valtest_data[:, 3].max()
            print("You can provide 'x' for any value to indicate a wildcard (any value).")
            subject_input = input("Please provide the subject_id (or subject-(sub)-string or x for wildcard): ")
            rel_input = input("Please provide the rel_id (or relation-(sub)-string or x for wildcard): ")
            object_input = input("Please provide the object_id (or object-(sub)-string or x for wildcard): ")
            timestep_input = input(f"Please provide the timestep - the range is <{timesteps_min},{timesteps_max}>: ")

            print(f'Your input was: {subject_input}, {rel_input}, {object_input}, {timestep_input}')

            quads = []
            quads = find_and_append_quads(subject_input, rel_input, object_input, timestep_input, quads, valtest_data, nodes_string_to_id, rels_string_to_id)

            for quad in quads:
                print(f"I will explain the quadruple: {quad[0]} {quad[1]} {quad[2]} {quad[3]}")
                subject_id = quad[0]
                rel_id = quad[1]
                object_id = quad[2]
                timestep = quad[3]
                try:
                    sub_string = dataset.nodes_id_to_string[subject_id]
                    rel_string = dataset.rels_id_to_string[rel_id]
                    obj_string = dataset.nodes_id_to_string[object_id]            
                except KeyError as e:
                    print("Please provide a valid quadruple.")
                    print(e, "not in the dataset.")
                print(f"{sub_string} {rel_string} {obj_string} {timestep}")
            quadruples = quads
            print(f"In total {len(quadruples)} quadruples will be explained.")

    ## check if the quadruples are in the test (or valid, dep on dataset_split) set
    all_quads_in_dataset =check_quads_in_testset(testset, dataset.nodes_id_to_string, dataset.rels_id_to_string, quadruples)
    if not all_quads_in_dataset:
        print("Please restart and provide a valid quadruple.")
        sys.exit(0)


    dataset.explain_data = quadruples
    for quad in quadruples:
        dataset.index(int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3]), "explain") 
    dataset.create_explain_gt_index(mode=dataset_split) #  create a dict that is used in explain mode.
        # It contains the ground truth objects for each query - to show the user
        #  which objects besides the test quadruples object are true at this time

    if len(quadruples) == 0:
        print('no quads found')
        print("Please restart and provide a valid quadruple.")
        sys.exit(0)
    return dataset, dataset_name, testset, rule_path, quadruples

def find_all_ids_with_string(input_string,  mapping):
    """ find all entries in the mapping, that at least partly contain the input_string.
    e.g. the input_string woman should return [mapping["woman"], mapping["WOMAN"], 
    mapping["woman(australia)"], mapping["all_woman"] ,...
    :param input_string: [string] string that partly describes node or relation of interest
    :param mapping: [dict] e.g. node_string_to_id_mapping, has as key the node or rel string representations and value the ids
    :return ids: [list] list with all ids that match the input string
    """
    ids = []
    for key in mapping:
        if input_string.lower() in key.lower(): # ignore case
            ids.append(mapping[key])
    return ids

def find_and_append_quads(subject_input, rel_input, object_input, timestep_input, quads, valtest_data, nodes_string_to_id, rels_string_to_id):
    """ for given inputs (subject_input, rel_input, object_input, timestep_input), transform to usable id-representation and append to quads
    multiple options:
    1) the ids are already given as int. then just return
    2) the ids are partly 'x'. then find all entries in valtest_data, that match the given ids
    3) the ids are strings, e.g. subject_input = woman. then find all entries in valtest_data with node_id_to_string_mapping that match
    :param subject_input: [int or string] depending on the three options above, the user input for a quadruple
    :param rel_input: [int or string] depending on the three options above, the user input for a quadruple
    :param object_input: [int or string] depending on the three options above, the user input for a quadruple
    :param timestep_input: [int or string] depending on the three options above, the user input for a quadruple
    :param quads: list with quadruples, each quadruple is a set (subject_id, rel_id, object_id, timestep)
    :param valtest_data: [np.array] either the validation or test data from dataset, i.e. dataset.val_data, or test_data, with each line (sub, rel, obj, timestamp)
    :param nodes_string_to_id: [dict], contains as key the node string (e.g. "einstein") and  value the id in the dataset
    :param rels_string_to_id: [dict], contains as key the rel string (e.g. "likes") and  value the id in the dataset
    :return quads: list with quadruples, each quadruple is a set (subject_id, rel_id, object_id, timestep)
    """
    
    try:
        # 1) the ids are already given as int. then just return
        subject_id = int(subject_input)
        rel_id = int(rel_input)
        object_id = int(object_input)
        timestep = int(timestep_input)
        quads.append((subject_id, rel_id, object_id, timestep))
        return quads
    except ValueError:

        # 2) the ids are partly 'x'. then find all entries in valtest_data, that match the given ids
        subject_ids = [subject_input]
        rel_ids = [rel_input]
        object_ids = [object_input]
        timestep = timestep_input
        if subject_input != 'x':
            try:
                subject_ids = [int(subject_input)]
            except ValueError:
                subject_ids = find_all_ids_with_string(subject_input, nodes_string_to_id)
        if rel_input != 'x':
            try:
                rel_ids = [int(rel_input)]
            except ValueError:
                rel_ids = find_all_ids_with_string(rel_input, rels_string_to_id)
        if object_input != 'x':
            try:
                object_ids = [int(object_input)]
            except ValueError:
                object_ids = find_all_ids_with_string(object_input, nodes_string_to_id)

        for subject_id in subject_ids:
            for rel_id in rel_ids:
                for object_id in object_ids:
                    for valtest_quad in valtest_data:
                        # 3) all the ids with 'x'
                        if (subject_id == 'x' or int(subject_id) == valtest_quad[0]) and \
                                (rel_id == 'x' or int(rel_id) == valtest_quad[1]) and \
                                (object_id == 'x' or int(object_id) == valtest_quad[2]) and \
                                (timestep == 'x' or int(timestep) == valtest_quad[3]):
                            quads.append((int(valtest_quad[0]), int(valtest_quad[1]), int(valtest_quad[2]), int(valtest_quad[3])))


    return quads

def read_quads(quadruples_file, valtest_data, nodes_string_to_id, rels_string_to_id):
    """ Read quadruples from a file. this works with ids, wildcards 'x', or node/rel strings
    :param quadruples_file: [str] path to the file containing the quadruples
    :param valtest_data: [np.array] either the validation or test data from dataset, i.e. dataset.val_data, or test_data, with each line (sub, rel, obj, timestamp)
    :param nodes_string_to_id: [dict], contains as key the node string (e.g. "einstein") and  value the id in the dataset
    :param rels_string_to_id: [dict], contains as key the rel string (e.g. "likes") and  value the id in the dataset
    :return quads: [list] list of quadruples each quadruple is a set [(subject_id, rel_id, object_id, timestep)]
    """
    quads = []
    with open(quadruples_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            if 'sub' in line:
                continue
            subject_input, rel_input, object_input, timestep = line.strip().split(' ')
            quads = find_and_append_quads(subject_input, rel_input, object_input, timestep, quads, valtest_data, nodes_string_to_id, rels_string_to_id)


    return quads

def check_quads_in_testset(testset, nodes_id_to_string, rels_id_to_string, quadruples):
    """ Check if the quadruples are in the test set.
    :param testset: [dict] test set containing the quadruples in form [subject_id][rel_id][object_id][timestep]
    :param nodes_id_to_string: [dict] mapping of node ids to node strings
    :param rels_id_to_string: [dict] mapping of relation ids to relation strings
    :param quadruples: [list] list of quadruples, each quadruple a set [(subject_id, rel_id, object_id, timestep)]
    :return: [bool] True if all quadruples are in the test set, False otherwise
    """
    all_quads_in_dataset = True
    for quad in quadruples:
        error_flag = False
        if quad[0] not in testset:
            print(f"Subject {quad[0]} not in the test set.")
            error_flag = True
        elif quad[1] not in testset[quad[0]]:
            print(f"Relation {quad[1]} not in the test set.")
            error_flag = True
        elif quad[2] not in testset[quad[0]][quad[1]]:
            print(f"Object {quad[2]} not in the test set.")
            error_flag = True
        elif quad[3] not in testset[quad[0]][quad[1]][quad[2]]:
            print(f"Timestep {quad[3]} not in the test set.")
            error_flag = True
        if error_flag:
            sub_string = nodes_id_to_string[quad[0]]
            rel_string = rels_id_to_string[quad[1]]
            obj_string = nodes_id_to_string[quad[2]]
            print(f"Quadruple {sub_string} {rel_string} {obj_string} {quad[3]} not in the set.")
            print("Please provide a valid quadruple.")
            all_quads_in_dataset = False
        
    return all_quads_in_dataset

def explain(dataset, out_path, dataset_split,  path_rules,options_explain, max_rules_per_pred, plot_figures_flag):

    # path_rankings = utils.get_path_rankings_name(out_path, 'ranks.txt', dataset.dataset.name, dataset_split, options) # this is the path where the computed rankings are written to
    path_rankings = os.path.join(out_path, 'ranks.txt')

    num_rules, rule_triple_dict = predictor.apply(dataset, path_rules, path_rankings, 0.01, evaluation_mode=dataset_split, explain_flag=True, options_explain=options_explain)
    
    write_explanations(out_path, rule_triple_dict, dataset, max_rules_per_pred, plot_figures_flag)
    return num_rules, rule_triple_dict


def write_explanations(out_path, rule_triple_dict, dataset, max_rules_per_pred, plot_figures_flag):
    outfile = os.path.join(out_path, 'explanations.txt')
    figure_path =   os.path.join(out_path, 'figures')

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    with open(outfile, 'w', encoding="utf-8") as file:
        quad_to_explain_list = []
        quad_to_pred_list =[]
        
        for sub in dataset.head_rel_tail_t['explain']:
            for rel in dataset.head_rel_tail_t['explain'][sub]:
                for obj in dataset.head_rel_tail_t['explain'][sub][rel]:
                    for t in dataset.head_rel_tail_t['explain'][sub][rel][obj]:
                        quad_to_explain_list.append((sub, rel, obj, t))

        for quad in quad_to_explain_list:
            query = (quad[0], quad[1], quad[3]) # sub, rel, t
            quad_preds = rule_triple_dict[query]
            
            # if query not in pred_per_quad:
            #     pred_per_quad[query] = {}
            quad_preds_sorted =dict(
                    sorted(
                        quad_preds.items(), # iterates over the key-value pairs of the outer dictionary.
                        key=lambda item: max(item[1].keys()),  # Extract the inner dictionary key # item[1].keys() Accesses the keys of the inner dictionary. # max(item[1].keys()): Gets the key of the inner dictionary, which is used as the sorting criterion.
                        reverse=True
                    )
                )
                            
            
            # {key: quad_preds[key] for key in sorted(quad_preds, reverse=True)}
            quad_string = f"{dataset.nodes_id_to_string[quad[0]]} {dataset.rels_id_to_string[quad[1]]} {dataset.nodes_id_to_string[quad[2]]} {quad[3]}"
            string0 = f"{quad[0]} {quad[1]} {quad[2]} {quad[3]}\t{dataset.nodes_id_to_string[quad[0]]} {dataset.rels_id_to_string[quad[1]]} {dataset.nodes_id_to_string[quad[2]]} {quad[3]}\n"
            string_gt = 'all gt nodes: '
            for tail in dataset.head_rel_t_tail['explain'][quad[0]][quad[1]][quad[3]]:
                string_gt += f"{tail} {dataset.nodes_id_to_string[tail]} "
            string_gt += '\n'
            file.write('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')
            file.write(string0)
            file.write(string_gt)
            # quad_preds.sort(key=lambda x: x[1], reverse=True) # sort by assigned scores
            index = 0
            rule_strings = []
            predicted_nodes_counter = 0
            score_dict_list = []
            for predicted_node, infos in quad_preds_sorted.items(): # which nodes are predicted for this trple
                if predicted_nodes_counter < max_rules_per_pred: # show only the ten highest nodes
                    for predicted_score, rules_infos in infos.items(): # which rules did predict the node
                        predicted_node_string = dataset.nodes_id_to_string[predicted_node][0]
                        predicted_score = round(predicted_score, 5)
                        string1 = f"{predicted_node}\t{predicted_node_string}\t{predicted_score}\n"
                        file.write(string1)
                        
                        rule_dict_list = []
                        for rule_info in rules_infos:
                            rule_line_dict = {}
                            rule_score = rule_info[0]
                            rule_ids = rule_info[1]
                            rule_node = rule_info[2]
                            rule_params = rule_info[3]
                            rule_single_freq = rule_info[5]
                            score_single = rule_info[6]
                            score_multi = rule_info[7]
                            p_rounded = []
                            for p in rule_params:
                                if type(p) == float:
                                    p_rounded.append(round(p, 4))
                                else:
                                    p_rounded.append(p)
                            # if rule_single_freq == 's':
                            #     p_rounded = p_rounded[0:3]
                            # else:
                            #     p_rounded = p_rounded[3:6]
                            p_rounded = p_rounded[0:6]
                            rule_params = tuple(p_rounded)

                            quad_firing_rule = '['
                            quad_index = 0
                            for i in rule_info[4]:
                                quad_firing_rule += str(i)
                                if quad_index == 3:
                                    quad_firing_rule += '] '
                                # if quad_index == 4:                            
                                #     quad_firing_rule += 'x'
                                quad_index +=1
                                quad_firing_rule+= ' '
                            # if rule_single_freq == 's':
                            #     quad_firing_rule += ' recency'
                            # else:
                            #     quad_firing_rule += ' frequency'

                            if len(rule_ids) <2: # z-rule
                                a =1
                                string_repre = 'F\tZ-rule: ' + dataset.rels_id_to_string[rule_info[4][1]] + '(X, ' + dataset.nodes_id_to_string[rule_info[4][2]][0] +', T)'
                                id_repre = 'F\tZ-rule: ' +str(rule_info[4][1]) + '(X, ' + str(rule_info[4][2])+', T)'
                                rule_key = None
                                rule_key2 = None
                            elif len(rule_ids) == 3: # f-rule (relh,  c_head, relb, c_body)
                                string_repre = 'F\tF-rule: ' + dataset.rels_id_to_string[rule_ids[0]] + '('+dataset.nodes_id_to_string[rule_ids[1]][0]+','+dataset.nodes_id_to_string[rule_ids[2]][0]+',T)'+' <= ex.' + dataset.rels_id_to_string[rule_ids[0]] + '('+dataset.nodes_id_to_string[rule_ids[1]][0]+',?,T)'
                                id_repre = 'F\tF-rule: ' + str(rule_ids[0]) + '('+str(rule_ids[1])+','+str(rule_ids[2])+',T)'+' <= ex.' + str(rule_ids[0]) + '('+str(rule_ids[1])+',?,T)'
                                # id_repre = 'f-rule, rel: ' +str(rule_info[4][1]) + ' obj: ' + str(rule_info[4][2])
                                rule_key = None
                                rule_key2 = None  
                            else: # Protest_violently,_riot(X,Y,T) <= inv_Use_tactics_of_violent_repression(X,Y,U
                                if len(rule_ids) == 2:
                                    rule = Rule2(rule_ids[0], rule_ids[1], rule_params, dataset.rels_id_to_string[rule_ids[0]],  dataset.rels_id_to_string[rule_ids[1]])
                                  
                                # (relh,  c_head, relb, c_body)
                                if len(rule_ids) == 4: # relh_string, ch_string, relb_string, cb_string):
                                    rule = RuleC(*rule_ids, rule_params, dataset.rels_id_to_string[rule_ids[0]], dataset.nodes_id_to_string[rule_ids[1]][0],
                                                dataset.rels_id_to_string[rule_ids[2]],
                                                dataset.nodes_id_to_string[rule_ids[3]][0])  #(self.relh, self.ch, self.relb, self.cb)
                                if len(rule_ids) == 5:
                                    rule = RuleCBackward(*rule_ids[0:4], rule_params, dataset.rels_id_to_string[rule_ids[0]], dataset.nodes_id_to_string[rule_ids[1]][0],
                                                dataset.rels_id_to_string[rule_ids[2]],
                                                dataset.nodes_id_to_string[rule_ids[3]][0]) 
                                string_repre = rule.get_string_repre()
                                id_repre = rule.get_id_repre()
                                rule_key = str(rule.get_rule_key())
                                rule_key2 = rule_key.replace(", ", "_").replace("(", "").replace(")", "")
                            
                            rule_score = round(rule_score, 5)
                            score_single = round(score_single,3)
                            score_multi = round(score_multi, 3)
                            num_firing = rule_info[4][4]
                            # all_firing_dists =  [int(quad[3]) - t for t in rule_info[4][5]]
                            string2 = f"{rule_score}\t({score_single}+{score_multi})\t{id_repre}\t{string_repre}\t{rule_params}\t{quad_firing_rule}\t{num_firing}\n"
                            if index == 0:
                                rule_strings.append(string2)
                            

                            file.write(string2)
                            
                            rule_line_dict['rule_score'] = rule_score
                            rule_line_dict['rule_single_score'] = score_single
                            rule_line_dict['rule_multi_score'] = score_multi
                            rule_line_dict['rule_ids'] = id_repre
                            rule_line_dict['rule_strings'] = string_repre
                            rule_line_dict['rule_params'] = rule_params
                            
                            rule_line_dict['quad'] = quad_firing_rule
                            rule_line_dict['num_firing'] = str(num_firing) + 'x '
                            if rule_info[4][3] > -1:
                                rule_line_dict['time_dist'] = int(quad[3]) - rule_info[4][3]
                            else:
                                rule_line_dict['time_dist'] = -1
                            # rule_line_dict['all_time_dists'] = [int(quad[3]) - t for t in rule_info[4][5]]
                            if rule_single_freq == 's':
                                rule_line_dict['image'] = f"rule{rule_key2}.png"
                            elif rule_single_freq == 'm':
                                rule_line_dict['image'] = f"rule{rule_key2}_singlemulti.png"

                            
                            rule_dict_list.append(rule_line_dict)
                            if plot_figures_flag:
                                if len(rule_ids) == 2: # if not z rules
                                    if not os.path.exists(os.path.join(figure_path, rule_line_dict['image'])):
                                        if rule_single_freq == 's':

                                            
                                            make_plot(rule_params, rule_line_dict['image'], figure_path, string_repre,  max_timestep=100)
                                        elif rule_single_freq == 'm':
                                            
                                            make_plot_multi(rule_params, rule_line_dict['image'], figure_path, string_repre, window=50)

                                
                        score_dict = {}
                        
                        score_dict['pred_id'] = predicted_node
                        score_dict['pred_string'] = predicted_node_string
                        score_dict['score'] = predicted_score
                        score_dict['rows'] = rule_dict_list
                        score_dict_list.append(score_dict)
                        #  write_html()
                        # if index == 0: # print only the highest node
                        #     if len(rule_triple_dict) < 6: # only print if we want less than 6 quads explained.
                        #         print_explanation(quad, quad_string, predicted_node=predicted_node, predicted_node_string=predicted_node_string, 
                        #                         highest_score=predicted_score, rule_id=id_repre, rule_strings=rule_strings)
                        index += 1
                predicted_nodes_counter +=1
            quad_to_pred_dict = {}
            quad_to_pred_dict['subject_id'] = quad[0]
            quad_to_pred_dict['rel_id'] = quad[1]
            quad_to_pred_dict['object_id'] = quad[2]
            quad_to_pred_dict['gt_nodes'] = dataset.head_rel_t_tail['explain'][quad[0]][quad[1]][quad[3]]
            quad_to_pred_dict['gt_nodes_string'] = [dataset.nodes_id_to_string[i][0] for i in quad_to_pred_dict['gt_nodes']]
            quad_to_pred_dict['timestep'] = quad[3]
            quad_to_pred_dict['subject_string'] = dataset.nodes_id_to_string[quad[0]][0]
            quad_to_pred_dict['rel_string'] = dataset.rels_id_to_string[quad[1]]
            quad_to_pred_dict['object_string'] = dataset.nodes_id_to_string[quad[2]][0]
            quad_to_pred_dict['timestep_string'] = quad[3]
            quad_to_pred_dict['pred_scores'] = score_dict_list
            quad_to_pred_list.append(quad_to_pred_dict)



        # predicted object, predicted score, rule, rule params

    

    write_html(quad_to_pred_list, output_folder=out_path)



def make_plot(rule_params, rule_fig_name, figure_path, rule_string, max_timestep=400):

    # ftype = 'powtwo'
    rule_params= tuple([*rule_params, 0,0,0,'powtwo'])


    plt.figure()
    x_gt = np.arange(1, max_timestep, 1)
    fct = score_single
    y_gt = fct(x_gt, rule_params, rule_params[-1]) 
    # y_gt = np.zeros(len(x_gt))
    plt.plot(x_gt, y_gt, label='rule curve') 

    plt.title(rule_string, fontsize=10)
    plt.xlabel('time distance')
    plt.ylabel('confidence')
    plt.grid()
    plt.legend()
    save_path = os.path.join(figure_path, rule_fig_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def make_plot_multi(rule_params, rule_fig_name, figure_path, rule_string,  window=50):
    # make two plots next to each other: one single and one that contains the devation for the props.

    # Create a figure with two subplots side by side
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(1, 2, figsize=(27, 6))
    
    # Set the title for the entire figure
    if 'X,Y' in rule_string or 'X, Y' in rule_string:
        rule_string = rule_string.split('\t', 1)[-1]
        # rule_string = rule_string.replace('X,Y', 'X, Y')
    fig.suptitle(rule_string, fontsize=16)
    # Set font size for all text in the plots
    
    # First subplot: Output of make_plot()
    rule_params = tuple([*rule_params, 0, 0, 0, 'powtwo'])

    x_gt = np.arange(1, 100, 1)
    fct = score_single
    lmbda, alpha, phi = rule_params[0:3]
    y_gt = fct(x_gt, lmbda, alpha, phi)
    axes[0].plot(x_gt, y_gt, label='rule curve')
    axes[0].set_title('recency score', fontsize=16)
    axes[0].set_xlabel('min time distance min($\Delta$)')
    axes[0].set_ylabel('confidence')
    axes[0].grid()
    axes[0].legend()

    # Second subplot: Empty figure (placeholder for now)
    time_window = 50

    x_values = np.arange(0, 1, 1 / time_window)
    fct_multi = score_linear
    y_values = []
    for x in x_values:
        y_values.append(fct_multi(x, *rule_params[3:6]))

    x_values *= time_window

    axes[1].plot(x_values[1:], y_values[1:], color='tan')
    
    axes[1].set_title(" frequency score (to be added)", fontsize=16)
    axes[1].set_xlabel(f"number of occurences within time window {time_window} for min($\Delta$) = 1")
    axes[1].set_ylabel("confidence difference")
    axes[1].grid()

    # Save the combined figure
    save_path = os.path.join(figure_path, rule_fig_name)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)





def print_explanation(test_query, quad_string, predicted_node, predicted_node_string, highest_score, rule_id, rule_strings):

    groundtruth = test_query[3]

    print(f"For the test query {test_query}, ({quad_string}), the model predicted")
    print(f"with the highest score {highest_score} the node {predicted_node} ({predicted_node_string}).")
    print("\nThe prediction was caused by (combining) the rule(s):")
    for rule_string in rule_strings:
        # rule_score, rule_id, rule_string, params, firing_quadruple = rule_string.split("\t")
        rule_score, forwardbackward, rule_id, forwardbackward2, rule_string, params, firing_quadruple = rule_string.split("\t")
        rule_id = rule_id.strip()
        rule_string = rule_string.strip()
        params = params.strip()
        firing_quadruple = firing_quadruple.strip()
        print(f"- {rule_id}\t{rule_string},")
        print(f"\t with score {rule_score}.")
        print(f"\t with parameters {params}." )
        print(f"\t The rule fired because of the quadruple {firing_quadruple}")



def get_gradient_color(score, min_score=1, max_score=300):
    """
    Berechnet eine Farbe aus einem Farbverlauf basierend auf einem Score.
    Der Farbverlauf reicht von '#1a4b66' # dark blue grey bis '#a3b8c4' # desaturated lightened version of dark blue grey.
    """
    color1 = '#1a4b66' # dark blue grey
    color2 = '#a3b8c4' # desaturated lightened version of dark blue grey

    # Convert hex to RGB
    start_color = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
    end_color = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))

    # Normalisiere den Score auf einen Wert zwischen 0 und 1
    normalized_score = np.log10(score) / np.log10(max_score + 1)
    # normalized_score = (score - min_score) / (max_score - min_score)
    normalized_score = max(0, min(1, normalized_score))  # Begrenze auf [0, 1]

    # Interpoliere die Farben
    r = int(start_color[0] + (end_color[0] - start_color[0]) * normalized_score)
    g = int(start_color[1] + (end_color[1] - start_color[1]) * normalized_score)
    b = int(start_color[2] + (end_color[2] - start_color[2]) * normalized_score)

    return f"rgb({r}, {g}, {b})"


def write_html(quads_dict, output_folder):

    # Generate HTML for each row
    rows_html = ""
    for quad in quads_dict:
        # write the quad        
        # quad_string = f"""
        # <div class="collapsible-container">
        # <button class="toggle-button" onclick="toggleContent(this)">▼</button>
        # <div class="container"> 
        # <h6><p><span class="ids"> {quad['subject_id']} {quad['rel_id']} </span><span class="pred"> {quad['object_id']} </span> <span class="ids">{quad['timestep']}
        #  <span class="str"> {quad['subject_string']} {quad['rel_string']} </span><span class="predstr"> {quad['object_string']}</span> <span class="str">{quad['timestep_string']} </span></p></h6>
        # </li> """

        # gt_string = f""" <h6><p><span class="predstr">Ground Truth Nodes:</span>                        
        #                 """
        # for gt_node, gt_node_string in zip(quad['gt_nodes'], quad['gt_nodes_string']):
        #     gt_string += f"""<span class="predstr"> {gt_node} {gt_node_string} </span>"""

        quad_string = f"""
        <div class="collapsible-container">
        <button class="toggle-button" onclick="toggleContent(this)">▼</button>
        <div class="container">
                <span class="verbose">
                <span class="ids">{quad['subject_id']} {quad['rel_id']}</span>
                <span class="pred">{quad['object_id']}</span>
                <span class="ids">{quad['timestep']}</span>
                </span>
                <span class="str">{quad['subject_string']} {quad['rel_string']}</span>
                <span class="predstr">{quad['object_string']}</span>
                <span class="str">{quad['timestep_string']}</span>

            <p><span class="predstr">Ground Truth Nodes:</span>
        """

        # Add GT nodes inline
        
        for gt_node in quad['gt_nodes']:
            quad_string += f"""<span class="verbose"> <span class="predstr"> {gt_node} </span> </span>"""
        for gt_node in quad['gt_nodes_string']:
            quad_string += f"""<span class="predstr"> {gt_node} </span>"""

        

        # Close paragraph and start list for predictions
        quad_string += """</p><ol class="pred-list">
        """
        # gt_string += f"""</p></h6></li>"""
        rows_html += quad_string
        # rows_html += gt_string
                        
        
        # write the scores for that quad  
        for pred_score in quad['pred_scores']:
            pred_string = f"""
                        <span class="verbose">{pred_score['pred_id']} </span> {pred_score['pred_string']} {pred_score['score']}
"""
            rows_html += f"""
                                <li><span class="verbose">{pred_score['pred_id']} </span> <span class="predstr"> {pred_score['pred_string']}</span><span class="score">	{pred_score['score']}</span></li>
                        """
            new_rows = pred_score['rows']
            rows_html += f"""<ol> """
                                    # <span style="color: ">{row['time_dist']}</span> 
            # write the rules for that score
            for row in new_rows:
                color = get_gradient_color(row['time_dist'])
                # rows_html += f"""<li>
                #                     <span class="score">{row['rule_score']}</span> ({row['rule_single_score']}+{row['rule_multi_score']}) {row['rule_ids']} {row['rule_strings']}
                #                     {row['rule_params']} {row['quad']} 
                #                     <span style="background-color: #e4eff1; color: {color}; padding: 2px 5px; border-radius: 3px; font-weight: bold;">  {row['time_dist']}</span> 
                #                     <a href="./figures/{row['image']}" target="_blank" class="image-link">
                #                         PLOT <img src="./figures/{row['image']}" alt="Image Preview" />
                #                     </a>
                #                 </li>
                #                 """
                rule_bf = row['rule_strings'].split('\t')[0]+'\t'
                rule_string = ' '.join(row['rule_strings'].split('\t')[1:])
                rows_html += f"""<li>
                    <span class="score">{row['rule_score']}</span> 
                    ({row['rule_single_score']}+{row['rule_multi_score']})
                    <span class="verbose">
                    {rule_bf}    
                    </span>
                    {rule_string}
                    <span class="verbose">
                        {row['rule_ids']} {row['rule_params']}
                    </span>
                    {row['num_firing']}
                    <span style="background-color: #e4eff1; color: {color}; padding: 2px 5px; border-radius: 3px; font-weight: bold;">
                        {row['time_dist']}
                    </span> 
                    <a href="./figures/{row['image']}" target="_blank" class="image-link">
                        PLOT <img src="./figures/{row['image']}" alt="Image Preview" />
                    </a>
                </li>"""
            rows_html += f"""</ol> """
        rows_html += f"""</ol>
         </div> """
        rows_html += f"""<p>	</p>
        </div>
        """

    # Replace the placeholder with the dynamically generated rows
    updated_html = html_template.replace("## PLACEHOLDER_PRED", rows_html)

    # Write the updated HTML to a file
    with open(os.path.join(output_folder,"explanations_fancy.html"), "w", encoding="utf-8") as file:
        file.write(updated_html)

    name = os.path.join(output_folder,"explanations_fancy.html")

    print(f"HTML file has been created with new rows: {name}")
    a=1


html_template = """ 
<!DOCTYPE html>
<html>
<head>
    <script>
    function toggleContent(button) {
        const container = button.parentElement;
        container.classList.toggle('expanded');
        button.classList.toggle('expanded');
    }
    </script>
   <link rel="stylesheet" href="styles.css">
</head>
<body>

<h3>Explanations</h3>

<!-- Hidden checkbox for verbose toggle -->
<input type="checkbox" id="verbose-toggle" hidden>
<label for="verbose-toggle" class="verbose-button"></label>

## PLACEHOLDER_PRED

</body>
</html>
"""