
import os.path as osp
import numpy as np
from matplotlib import pyplot as plt
from decimal import *
from rules import RuleSet, Rule1, Rule2, RuleC, RuleCBackward
import matplotlib.colors as mcolors


def make_packages(dists, betas, weights, positives, pfactors, max_time_window, rule_unseen_neg=0):
    """ data preprocessing for the multi time window learning
    make the packages for the multi time window learning
    each package belongs to one min_time_distance, and contains all the proportions of available timesteps, the aggregated beta and the aggregated weight
    
    :param dists: list of sets, inner set has ints, the time differences between the query and the occurences, e.g. [(1,), (2,1), (3,2)] WE ASSUME DISTS ARE SORTED IN DECREASING MANNER!
    :param betas: list of floats, the beta values for the given time differences
    :param weights: list of floats, the weights for the given time differences
    :param positives: lift of floats, the positives for the given time differences
    :param pfactors: list of floats, how is the num_occurences/(num_occurences+pvalue), i.e. how much does the pvalue affect a positive example per distance
    :param max_time_window: int, the maximum time window to be taken into account when computing the proportion of available time steps
    :param rule_unseen_neg: float param for laplace smoothing


    :return nonaggregated_package_dict: [dict] for each min distance (recency distance) we have a dictionary with the following entries. The entries are not aggregated. this means,
    we can have multiple entries for one dist with e.g. proportion 0.05. this could e.g. come from dists [1,2,4,5,7] and dists [1,2,3,4,5]
    - val: list, contains the distance values (e.g. (1,2,), (1,3,4), ...)
    - proportion: list, contains the proportion of available time steps in time window (e.g. 0.05)
    - beta: list, contains the beta values for the given time differences. here, the p-value is NOT applied
    - weight: list, contains the weights, i.e. number of occurences with given distance and given proportion for this specific entry
    - pfactor: list, how is the num_occurences/(num_occurences+pvalue), i.e. how much does the pvalue affect a positive example for this given distance
    
    :return agg_vals: [dict] for each min distance (recency distance) we have a tuple with the aggregated values for the given distance. this is equivalent to the content in learn_data single
    format. this is used to find the recency params. entries are
    (beta_agg, weight_agg) after adding the p-value
    """
    # 1) collect all the infos we have for each min distance in nonaggregated_package_dict
        # for each  min distance, e.g. 1 or 2 or 3..: 
        # entries for each "val" (e.g. (1,2,), (1,3,4), ...)
        # entries for each "proportion", i.e. how at many timesteps our of max_time_window did it fire (e.g. 0.5, 0.25, ...)
        # entries for each "beta" (e.g. 0.5, 0.6, ...)
        # entries for each "weight" (e.g. 183, 12, ...)
        # entries for each "pfactor" (e.g. 0.5, 0.6, ...) - this should be the same if the same distance is given
    
    nonaggregated_package_dict = {} # die brauch ich um multi zu berechnen
    index = 0
    for d in dists:
        dmin = d[-1] # we assume d is sorted in decreasing manner! 
        t_array = np.array(d) 
        within_max_time = t_array[t_array <= max_time_window] # Find elements within max_time
        if not dmin in nonaggregated_package_dict:
            if len(within_max_time) > 0:
                nonaggregated_package_dict[dmin] = {}
                nonaggregated_package_dict[dmin]['val'] = [d]                     
                nonaggregated_package_dict[dmin]['proportion'] = [len(within_max_time)/max_time_window]
                nonaggregated_package_dict[dmin]['beta'] = [betas[index]]
                nonaggregated_package_dict[dmin]['positives'] = [positives[index]]
                nonaggregated_package_dict[dmin]['weight'] = [weights[index]]
                # nonaggregated_package_dict[dmin]['pfactor'] = [pfactors[index]]
        else:
            if len(within_max_time) > 0:
                nonaggregated_package_dict[dmin]['val'].append(d)
                nonaggregated_package_dict[dmin]['proportion'].append(len(within_max_time)/max_time_window)
                nonaggregated_package_dict[dmin]['beta'].append(betas[index])
                nonaggregated_package_dict[dmin]['positives'].append(positives[index])
                nonaggregated_package_dict[dmin]['weight'].append(weights[index])
                # nonaggregated_package_dict[dmin]['pfactor'].append(pfactors[index])
        index +=1

    # 2) sort the values for each min distance; e.g.: if we have multiple entries with the same proportions: weighted mean. this is used for computing the single params
        # for each min distance, e.g. 1 or 2 or 3..:
        # aggregate beta: sum_i (beta_i*weight_i/sum(weight_i)) for i in all entries with the same proportion
        # aggregate weight: sum_i (weight_i) for i in all entries with the same proportion
        # apply p-factor to aggregated beta

    agg_vals = {}
    p_factors = {}

    for dmin in nonaggregated_package_dict:
        agg_vals[dmin] = (0,0) # this will contain for each min_distance the aggregated beta and the aggregated weight (number of occurences)
        index = 0
        for prop in nonaggregated_package_dict[dmin]['proportion']:
            weight_agg = agg_vals[dmin][1] +nonaggregated_package_dict[dmin]['weight'][index]
            beta_agg = (agg_vals[dmin][0]*agg_vals[dmin][1] +  nonaggregated_package_dict[dmin]['beta'][index]*nonaggregated_package_dict[dmin]['weight'][index]) /(weight_agg) # weighted mean beta

            agg_vals[dmin] = (beta_agg, weight_agg)
        
            index  +=1             

        agg_vals[dmin] = (beta_agg*weight_agg/(weight_agg+rule_unseen_neg), weight_agg) # add rule_unseen_neg 
        p_factors[dmin] = weight_agg/(weight_agg+rule_unseen_neg) # 



    return nonaggregated_package_dict,  agg_vals, p_factors


def read_id_to_string_mappings(root):
    """
    Reads the node_mapping and rel_mapping csv files and returns the mapping dictionaries.

    Args:
        root (str): The directory containing the node_mapping.csv and rel_mapping.csv files.
            The directory should be the same as the one containing the edgelist.csv file.
    
    Returns:
        two dictionaries - node_mapping_dict and rel_mapping_dict.
            node_mapping_dict (dict): A dictionary mapping node IDs to their corresponding labels and indices.
            the key is the id as used in the code, the values are a list [string, original_idx as in edgelist.csv]
            rel_mapping_dict (dict): A dictionary mapping relation IDs to their corresponding labels.
            the key is the id as used in the code and in edgelist.csv, the value is the string representation of the relation
    """
    # load the mapping csv for the node_mapping.csv and the rel_mapping.csv
    node_mapping = osp.join(root, 'node_mapping.csv')
    rel_mapping = osp.join(root, 'rel_mapping.csv')

    node_mapping_dict = {}
    rel_mapping_dict = {}
    with open(node_mapping, 'r', encoding='utf-8') as f:
        for line in f:
            if not 'original_id' in line:
                line = line.strip().split(';') 
                try:       
                    node_mapping_dict[int(line[1])] = [line[2], int(line[0])]
                except:
                    node_mapping_dict[int(line[1])] = ['todo', int(line[0])]
    with open(rel_mapping, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for line in f)
        num_rels = int(num_lines - 1)
        print('num_rels: ', num_rels)
    with open(rel_mapping, 'r', encoding='utf-8') as f:
        for line in f:
            if not 'original_id' in line:
                line = line.strip().split(';')        
                rel_mapping_dict[int(line[0])] = line[1]
                rel_mapping_dict[int(line[0])+num_rels] = 'inv_'+line[1]
    return node_mapping_dict, rel_mapping_dict

def read_rankings(path_rankings):
    '''
    Returns the rankings that are saved before in a text file
    :param path_rankings: the path of the file where the rankings should be loaded from
    :return rankings: rankings is a dictionary of the following format rankings[(sub, rel, obj, t)][candidate] = score of the candidate
    '''
    rankings = {}

    with open(path_rankings, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            sub, rel, obj, t = lines[i].split(" ")
            tuples_candidates_with_scores = {}
            if lines[i+1] != "\n":
                candidates = lines[i+1].split(" ")
                i = 0
                tuples_candidates_with_scores = {int(candidates[x]): float(candidates[x+1]) for x in range(0, len(candidates), 2)}

            rankings[(np.int64(sub), np.int64(rel), np.int64(t))] = tuples_candidates_with_scores
    return rankings


def read_rankings_random(path_rankings, num_nodes):
    '''
    Returns the rankings that are saved before in a text randomly sort the ties
    Instead of dircetly using the scores, we add 
    * randomly shuffle the order of the candidates with the same score
    * add an order to break ties, i.e. the first element in the list has the highest score, second element the second highest score, etc.
    :param path_rankings: the path of the file where the rankings should be loaded from
    :return rankings: rankings is a dictionary of the following format rankings[(sub, rel, obj, t)][candidate] = score of the candidate
    '''
    rankings = {}
    scores = np.linspace(1, 1/num_nodes, num_nodes).tolist() # we need maximum as many scores as we have nodes
    with open(path_rankings, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            sub, rel, _, t = lines[i].split(" ")
            tuples_candidates_with_scores = {}

            if lines[i+1] != "\n":
                candidates = lines[i+1].split(" ")

                tokens = [(int(candidates[x]), float(candidates[x+1])) for x in range(0, len(candidates), 2)] # candidate and score

                rankings_shuffled = []
                tie_rankings = []
                latest_score= 0.0

                for node_score in tokens:
                    if not(node_score[1] == latest_score) and latest_score != 0.0: # at the end of same score tokens                        
                        np.random.shuffle(tie_rankings) # shuffle the tie groups
                        for sr in tie_rankings:
                            rankings_shuffled.append(sr)
                        tie_rankings = []
                    tie_rankings.append(node_score) # collect all tokens with the same score (tie groups)
                    latest_score = node_score[1]

                np.random.shuffle(tie_rankings) # shuffle the same score tokens

                for sr in tie_rankings:
                    rankings_shuffled.append(sr) # append all the tie rankings to a list

                # we add an order to break ties, i.e. the first element in the list has the highest score, 
                # second element the second highest score, etc.
                cou = 0
                for x in range(0, len(rankings_shuffled)):
                    tuples_candidates_with_scores[int(rankings_shuffled[x][0])] = float(scores[cou]) 
                    cou+=1

            rankings[(np.int64(sub), np.int64(rel), np.int64(t))] = tuples_candidates_with_scores
    return rankings

def read_rankings_order(path_rankings, num_nodes):
    '''
    Returns the rankings that are saved before in a text file
    Instead of dircetly using the scores, we add an order to break ties, i.e. the first element in the list has the highest score, second element the second highest score, etc.
    :param path_rankings: the path of the file where the rankings should be loaded from
    :param num_nodes: [int] the number of nodes in the graph
    :return rankings: rankings is a dictionary of the following format rankings[(sub, rel, obj, t)][candidate] = score of the candidate
    '''
    rankings = {}
    scores = np.linspace(1, 1/num_nodes, num_nodes).tolist() # we need maximum as many scores as we have nodes

    with open(path_rankings, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            sub, rel, obj, t = lines[i].split(" ")
            tuples_candidates_with_scores = {}
            if lines[i+1] != "\n":
                candidates = lines[i+1].split(" ")
                i = 0
                for x in range(0, len(candidates), 2):
                    tuples_candidates_with_scores[int(candidates[x])] = float(scores[i]) 
                    i+=1
            rankings[(np.int64(sub), np.int64(rel),  np.int64(t))] = tuples_candidates_with_scores
    return rankings

# mannis code    
def read_rankings_as_list(path_rankings, k = 1000):
    '''
    Returns the rankings that are saved before in a text file.
    Uses a different data strucure compared to the other methods for reading rankings.
    A dictionary maps a key (sub, rel, obj, t) to a lists of pairs, where each pair
    consists of a candidate (int) and its confidence score (float).
    List l1 contains the ids of the candidates
    List l2 contains the corresponding scores.
    This datastructure keeps the order of the candidates as it finds them in the file.
    It also allows o retrieve the 7th candidate in constant time.
    :param path_rankings: the path of the file where the rankings should be loaded from
    :param max_candidates: the maximal number of candidates / scores stored in the list of candidates
    :return rankings: rankings as described above.
    '''
    rankings = {}
    with open(path_rankings, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            sub, rel, obj, t = lines[i].split(" ")
            scored_candidates = []
            if lines[i+1] != "\n":
                entries = lines[i+1].split(" ")
                n = 0
                for x in range(0, len(entries), 2):
                    scored_candidates.append((int(entries[x]), float(entries[x+1])))
                    n += 1
                    if n == k: break
            rankings[(int(sub), int(rel),  int(t))] = scored_candidates
    return rankings



def create_scores_array(predictions_dict, num_nodes):
    """ 
    Create an array of scores from a dictionary of predictions.
    predictions_dict: a dictionary mapping indices to values
    num_nodes: the size of the array
    returns: predictions an array of scores of len num_nodes
    """
    keys_array = np.fromiter(predictions_dict.keys(), dtype=int)
    values_array = np.fromiter(predictions_dict.values(), dtype=np.float64)
    
    predictions = np.zeros(num_nodes, dtype=np.float64)
    predictions[keys_array] = values_array
 
    return predictions



def fade_away_factor(delta_t, window_size_freq):
    fade_flag= False
    if fade_flag:
        fade_decay = -1/(window_size_freq-2) 
        fade_offset = (window_size_freq-1)/(window_size_freq-2)
        fade_factor = max(fade_decay* delta_t +fade_offset,0) # no negative fade factor in case outside of window
    else:
        fade_factor= 1
    return fade_factor

def linear(x, rho, kappa):
    return rho*x + kappa

def score_linear(proportion, rho, kappa, gamma):
    """ calculate the score using a linear function
    :param proportion: float, the proportion of elements within the time window
    :param rho: float, the rho parameter (slope)
    :param kappa: float, the kappa parameter (offset up down)
    :param gamma: float, the max y-value (clip)
    :return: float, the score for the given parameters
    """
    if proportion == 0:
        return 0
    val = linear(proportion, rho, kappa)
    val = np.clip(val, -gamma, gamma)
    return val


def score_linear_shift(proportion, dist, rho, kappa, gamma):
    """ calculate the score using a linear function
    section 4. equation (13)
    :param proportion: float, the proportion of elements within the time window
    :param rho: float, the rho parameter (slope)
    :param kappa: float, the kappa parameter (offset up down)
    :param gamma: float, the max y-value (clip)
    :return: float, the score for the given parameters
    """
    if proportion == 0:
        return 0
    val = rho*proportion + kappa/dist
    val = np.clip(val, -gamma, gamma)
    return val

def score_powtwo(delta_t, lmbda, phi):
    """ calculate the score using the two to the power of lambda * deltat function from history repeats itself paper

    :param delta_t: float, the time difference between the query and most recent occurence
    :param lmbda: float, the lambda parameter i.e. time decay
    :param alpha: float, the alpha parameter, i.e. rule confidence scale
    :param phi: float, offset parameter, to shift the function up and down (but still stay between 0 and 1)   
    :return: float, the score for the given parameters
    """
    offset = phi
    return 1.0/(1.0+offset)*(pow(2, -lmbda * (delta_t-1)) + offset)

def score_single(distance, lmbda, alpha, phi):
    """ Calculate candidate score depending on the time differences of the observed triples.
    section 4. equation (12)
    :param distance The distance for which the score is computed
    :param params: (lmbda, alpha, phi)
    :param function_type (str): type of function to use for scoring. Options: 'powtwo', 'increasing_powtwo', 'cos'
    :return: score (float): score for each sample computed using the given function
    """
    return alpha*score_powtwo(distance,lmbda,phi)
    

def write_rule_files(path_rules, ruleset): 
    """
    This function generates the rules files (it is temporary because we use only powto and the confidence).
    :param path_rules: the path of the file where the rules should be written to
    :param ruleset: the needed ruleset to be written in a file 
    """

    ruleset.write_ids(path_rules.format("ids"))
    ruleset.write_strings(path_rules.format("strings"))
    print("write rules to: ", path_rules.format("ids"), " and ", path_rules.format("strings"))

def generate_ruleset(dataset, options):
    """ generates the rulset based on the given flags -  with empty parameters
    :param dataset: the needed dataset object for data preprocessing
    :return ruleset: object of class ruleset
    """             
    reccuring_flag = options["RULE_TYPE_CYC1_REC"]
    cyclic_flag = options["RULE_TYPE_CYC1_NON_REC"]  
    ruleset = RuleSet(dataset.rels_id_to_string, dataset.nodes_id_to_string)

    for relb in dataset.rels_set:
        for relh in dataset.rels_set:
            if reccuring_flag and relb == relh:
                rule = Rule1(relb, [], dataset.rels_id_to_string[relb])
                ruleset.add_rule(rule)
            if cyclic_flag and relb != relh:
                rule = Rule2(relh, relb, [], dataset.rels_id_to_string[relh], dataset.rels_id_to_string[relb])
                ruleset.add_rule(rule)
    return ruleset

def extend_ruleset_constants(ruleset, learn_input, dataset, options): 
    """ extend the ruleset with rules with constants. We do not extend it by all possible rules with constants, but only by those that have been generated in the learn_input,
    i.e. those that have appearances > min threshold
    :param ruleset: the ruleset to be extended
    :param learn_input: dict. the learn input that contains all rules. key: rulekey (len 2 for Rule1,Rule2, len 4 for RuleC.)
    :param dataset: the needed dataset object for data preprocessing
    :return ruleset: the extended ruleset, object class
    """
    num_rules_added = 0
    num_backward_rules_added = 0
    for rule_key in learn_input:
        if len(rule_key) == 4:
            relh = rule_key[0]
            ch = rule_key[1]
            relb = rule_key[2]
            cb = rule_key[3]
            rule = RuleC(relh, ch, relb, cb, [], dataset.rels_id_to_string[relh], dataset.nodes_id_to_string[ch], dataset.rels_id_to_string[relb], dataset.nodes_id_to_string[cb])
            ruleset.add_rule(rule)
            num_rules_added += 1
        if len(rule_key) == 5:
            relh = rule_key[0]
            ch = rule_key[1]
            relb = rule_key[2]
            cb = rule_key[3]
            rule = RuleCBackward(relh, ch, relb, cb, [], dataset.rels_id_to_string[relh], dataset.nodes_id_to_string[ch], dataset.rels_id_to_string[relb], dataset.nodes_id_to_string[cb])
            ruleset.add_rule(rule)
            num_rules_added +=1
            num_backward_rules_added += 1
    if num_rules_added == 0: print("We added zero rules with constants. This might be due to a problem with the learn_input. Please double-check, you might have created your learn_input without considering the rules w. constants.")
    # print(f"we added {num_backward_rules_added} backward rules with constants")
    # print(f"we added {num_rules_added - num_backward_rules_added} forward rules with constants")
    return ruleset

def compute_error(params, X, y_gt, fct, sigma=1):
    """ compute the mean squared error of the prediction given the prediction function and the parameters of the fitted curve
    :param params: np.array, the parameters of the fitted curve, lmbda, alpha, phi
    :param X: np.array, the x values of the ground truth data, i.e. delta_t
    :param y_gt: np.array, the y values of the ground truth data, i.e. beta_hat
    :param fct: function, the function that was used to fit the curve
    :param sigma: None or scalar or M-length sequence, optional. Determines the uncertainty in ydata, or in our case the weight of each element in y-data. 
    If we define residuals as r = ydata - f(xdata, *popt), then the interpretation of sigma is: contains values of standard deviations of errors in ydata. I
    n this case, the optimized function is chisq = sum((r / sigma) ** 2). sigma = 1/sqrt(num_occurences) makes it the weight of each data point
    :return: float, the mean squared error of the prediction
    """
    y_pred = np.zeros(len(X))
    for i, x in enumerate(X):
        y_pred[i] = fct(x, *params)
    # mse = np.mean(((y_gt - y_pred)/sigma)**2)

    mse = np.sum(((y_gt - y_pred)/sigma)**2)/np.sum((1/sigma)**2) # sigma = 1/sqrt(num_occurences_of_this_datapoint), thus np.sum(np.sqrt(1/sigma)) is equal to the number of data points
    return mse





def plot_multi2(params, nonaggregated_package_dict, p_factors_dict, rul, figure_path, node_mapping_dict, rel_mapping_dict, max_time_window):
    """ plot figures for the new 2025 multi. still in its beginning, but will be extended
    """

    # stuff needed for the figure title
    backwardforward = 'F'
    if len(rul) > 2:
        rule_body = rul[2]
        rule_body_const = rul[3]
        rule_head = rul[0]
        rule_head_const = rul[1]
        if rule_head_const in node_mapping_dict:
            rule_head_const_st = node_mapping_dict[rule_head_const][0]
            rule_body_const_st = node_mapping_dict[rule_body_const][0]
        else:
            rule_head_const_st = str(rule_head_const)
            rule_body_const_st = str(rule_body_const)
        if len(rul) > 4:
            backwardforward = 'B'
    else:
        rule_body = rul[0]
        rule_head = rul[1]
        rule_body_const = 'Y'
        rule_head_const = 'Y'
        rule_head_const_st = 'Y'
        rule_body_const_st = 'Y'

    # computing the multi-correction factor for all possible proportions, based on given params
    props_possible = np.arange(1/max_time_window, 1, 1/max_time_window)
    y_pred_multi = {}
    for prop in props_possible:
        y_pred_multi[round(float(prop),4)] = score_linear(prop, params[3], params[4], params[5])

    # preparing the data for the plot
    X_list_single = []
    X_list_multi = []
    X_gt_list_multi = []
    y_gt_list_multi = []
    y_gt_weighted_mean = []
    y_pred_list_single = []
    y_pred_list_multi = []
    x_weighted_mean = []
    y_pred_weighted_mean = []
    y_gt_weighted_mean_before_p = []
    weight_list_multi = []
    y_pred_multi_shift = []
    for min_dist, min_dist_package in nonaggregated_package_dict.items():
        if min_dist >50:  # do only plot until 10, to not clutter the plot
            continue

        props_dict = {}
        index = 0
        
        # data processing to get the ground truth multi data; we need to aggregate the multiple proportion values; e.g. (1,2) and (1,3) should be aggregated to distance 1 and propotion 2/window_size
        for prop in min_dist_package['proportion']:
            weight = min_dist_package['weight'][index]
            beta_orig = min_dist_package['beta'][index]
            beta_afterp = min_dist_package['beta'][index]*p_factors_dict[min_dist]

            beta_orig_w = beta_orig*weight
            beta_afterp_w = beta_afterp*weight

            if prop not in props_dict:
                props_dict[prop] = [weight, beta_orig_w, beta_afterp_w, prop]
            else:
                props_dict[prop][0] += weight
                props_dict[prop][1] += beta_orig_w
                props_dict[prop][2] += beta_afterp_w
                
            index +=1

        for prop in props_dict:
            props_dict[prop][1] = props_dict[prop][1]/props_dict[prop][0]
            props_dict[prop][2] = props_dict[prop][2]/props_dict[prop][0]


        # single score for this distance
        x_single = min_dist
        props_dist = [x[3] for x in props_dict.values()]
        x_multi =  x_single* np.ones(len(props_dist)) + props_dist
        # x_multi = x_single* np.ones(len(props_possible)) + props_possible
        y_pred_single = score_single(min_dist, *params[0:3])

        y_pred_multi_shift = []
        for prop in props_dist:
            y_pred_multi_shift.append(score_linear_shift(prop, min_dist, *params[3:6])) # score_linear_shift(proportion, dist, rho, kappa, gamma, lmbda_exp):

        # multi score for this distance: single score plus multi correction factor
        y_pred_multi_dist = y_pred_multi_shift # [y_pred_multi[x] for x in props_dist]
        y_pred = y_pred_single*np.ones(len(y_pred_multi_dist)) + y_pred_multi_dist
        y_pred = np.clip(y_pred, 0, 1) # for now: clip the values to 0 and 1 until i have figured out gamma thing

        # ground truth values for the multi score
        x_gt_multi = min_dist*np.ones(len(props_dict)) + list(props_dict.keys()) # fake x values for the ground truth data
        y_gt_multi = [x[2] for x in props_dict.values()] # confidence values after being multiplied with pfactor
        y_gt_multi_before_p = [x[1] for x in props_dict.values()] # confidence values before being multiplied with pfactor
        y_weights_multi = [x[0] for x in props_dict.values()] # weights of the ground truth data

        # append the data to the lists
        X_list_single.append(x_single)
        X_list_single.append(x_single+1)
        
        X_list_multi += list(x_multi)
        y_pred_list_single.append(y_pred_single)
        y_pred_list_single.append(y_pred_single)
        y_pred_list_multi += list(y_pred)     
        X_gt_list_multi += list(x_gt_multi)
        y_gt_list_multi  += list(y_gt_multi)
        weight_list_multi += list(y_weights_multi) # could be used for coloring the points

        y_gt_weighted_mean.append(np.sum(np.array(y_gt_multi)*np.array(y_weights_multi))/np.sum(y_weights_multi))
        y_gt_weighted_mean.append(np.sum(np.array(y_gt_multi)*np.array(y_weights_multi))/np.sum(y_weights_multi))

        y_gt_weighted_mean_before_p.append(np.sum(np.array(y_gt_multi_before_p)*np.array(y_weights_multi))/np.sum(y_weights_multi))
        y_gt_weighted_mean_before_p.append(np.sum(np.array(y_gt_multi_before_p)*np.array(y_weights_multi))/np.sum(y_weights_multi))

        y_pred_weighted_mean.append(np.sum(np.array(y_pred)*np.array(y_weights_multi))/np.sum(y_weights_multi))
        y_pred_weighted_mean.append(np.sum(np.array(y_pred)*np.array(y_weights_multi))/np.sum(y_weights_multi))
        x_weighted_mean.append(min_dist)
        x_weighted_mean.append(min_dist+1)


    plt.figure(figsize=(80, 10))

    norm = mcolors.LogNorm(vmin=1, vmax=max(y_weights_multi))
    plt.plot(X_list_single, y_pred_list_single, '--',
            label='fitted curve %s: lmbd=%5.3f, alpha=%5.3f, phi=%5.3f ' % ('powtwo', *tuple(params[0:3])))  
    plt.plot(x_weighted_mean, y_gt_weighted_mean_before_p, alpha=0.5, c='lightblue', label='weighted mean ground truth data - before single p-value')
    plt.plot(x_weighted_mean, y_gt_weighted_mean, c='blue', alpha=0.5, label='weighted mean ground truth data - after single p-value')
    plt.plot(x_weighted_mean, y_pred_weighted_mean, c='darkred', alpha=0.5, label='weighted mean predictions')
    plt.scatter(X_list_multi, y_pred_list_multi, c='darkred', s=10, alpha=0.3, label='multi predictions NEW')
    # plt.scatter(X_list_multi, y_pred_multi_shift, c='grey', s=10, alpha=0.5, label='multi predictions NEW')
    plt.scatter(X_gt_list_multi, y_gt_list_multi, c=weight_list_multi,  s=10,  label='multi ground truth',cmap='viridis',norm=norm)

    plt.colorbar() # color scale to show the counts for each value

    plt.title(rel_mapping_dict[rule_head] + "(X," + rule_head_const_st + ",T)"  +' <=  '+rel_mapping_dict[rule_body]+"(X," + rule_body_const_st + ",U)", fontsize=10)
    plt.xlabel('time distance')
    plt.ylabel('beta')

    plt.legend()
    # Add secondary ticks TODO
    # ax = plt.gca()  # Get the current axis
    # ax2 = ax.twiny()  # Create a new twin x-axis sharing the same y-axis
    # ax2.set_xticks(min_dist_list)  # Set the locations for secondary ticks
    # ax2.set_xticklabels(min_dist_labels, fontsize=6, rotation=90)
    # ax2.set_xlim(ax.get_xlim())  # Match the x-limits of the primary axis
    plt.grid()
    plt.savefig(figure_path+'/multi_'+str(rule_body)+'-'+str(rule_body_const)+'-'+str(rule_head)+'-'+str(rule_head_const)+'-'+backwardforward+'.png', bbox_inches='tight')
    plt.close('all')


def plot_multi(params, nonaggregated_package_dict, p_factors_dict, rul, figure_path, node_mapping_dict, rel_mapping_dict, max_time_window):
    """ plot figures for the new 2025 multi. still in its beginning, but will be extended
    """

    # stuff needed for the figure title
    backwardforward = 'F'
    if len(rul) > 2:
        rule_body = rul[2]
        rule_body_const = rul[3]
        rule_head = rul[0]
        rule_head_const = rul[1]
        if rule_head_const in node_mapping_dict:
            rule_head_const_st = node_mapping_dict[rule_head_const][0]
            rule_body_const_st = node_mapping_dict[rule_body_const][0]
        else:
            rule_head_const_st = str(rule_head_const)
            rule_body_const_st = str(rule_body_const)
        if len(rul) > 4:
            backwardforward = 'B'
    else:
        rule_body = rul[0]
        rule_head = rul[1]
        rule_body_const = 'Y'
        rule_head_const = 'Y'
        rule_head_const_st = 'Y'
        rule_body_const_st = 'Y'

    # computing the multi-correction factor for all possible proportions, based on given params
    props_possible = np.arange(1/max_time_window, 1, 1/max_time_window)
    y_pred_multi = {}
    for prop in props_possible:
        y_pred_multi[round(float(prop),4)] = score_linear(prop, params[3], params[4], params[5])

    # preparing the data for the plot
    X_list_single = []
    X_list_multi = []
    X_gt_list_multi = []
    y_gt_list_multi = []
    y_gt_weighted_mean = []
    y_pred_list_single = []
    y_pred_list_multi = []
    x_weighted_mean = []
    y_pred_weighted_mean = []
    y_gt_weighted_mean_before_p = []
    weight_list_multi = []
    for min_dist, min_dist_package in nonaggregated_package_dict.items():
        if min_dist >50:  # do only plot until 10, to not clutter the plot
            continue

        props_dict = {}
        index = 0
        
        # data processing to get the ground truth multi data; we need to aggregate the multiple proportion values; e.g. (1,2) and (1,3) should be aggregated to distance 1 and propotion 2/window_size
        for prop in min_dist_package['proportion']:
            weight = min_dist_package['weight'][index]
            beta_orig = min_dist_package['beta'][index]
            beta_afterp = min_dist_package['beta'][index]*p_factors_dict[min_dist]

            beta_orig_w = beta_orig*weight
            beta_afterp_w = beta_afterp*weight

            if prop not in props_dict:
                props_dict[prop] = [weight, beta_orig_w, beta_afterp_w, prop]
            else:
                props_dict[prop][0] += weight
                props_dict[prop][1] += beta_orig_w
                props_dict[prop][2] += beta_afterp_w
                
            index +=1

        for prop in props_dict:
            props_dict[prop][1] = props_dict[prop][1]/props_dict[prop][0]
            props_dict[prop][2] = props_dict[prop][2]/props_dict[prop][0]


        # single score for this distance
        x_single = min_dist
        props_dist = [x[3] for x in props_dict.values()]
        x_multi =  x_single* np.ones(len(props_dist)) + props_dist
        # x_multi = x_single* np.ones(len(props_possible)) + props_possible
        y_pred_single = score_single(min_dist, *params[0:3])

        # multi score for this distance: single score plus multi correction factor
        y_pred_multi_dist = [y_pred_multi[x] for x in props_dist]
        y_pred = y_pred_single + y_pred_multi_dist
        y_pred = np.clip(y_pred, 0, 1) # for now: clip the values to 0 and 1 until i have figured out gamma thing

        # ground truth values for the multi score
        x_gt_multi = min_dist*np.ones(len(props_dict)) + list(props_dict.keys()) # fake x values for the ground truth data
        y_gt_multi = [x[2] for x in props_dict.values()] # confidence values after being multiplied with pfactor
        y_gt_multi_before_p = [x[1] for x in props_dict.values()] # confidence values before being multiplied with pfactor
        y_weights_multi = [x[0] for x in props_dict.values()] # weights of the ground truth data

        # append the data to the lists
        X_list_single.append(x_single)
        X_list_single.append(x_single+1)
        
        X_list_multi += list(x_multi)
        y_pred_list_single.append(y_pred_single)
        y_pred_list_single.append(y_pred_single)
        y_pred_list_multi += list(y_pred)     
        X_gt_list_multi += list(x_gt_multi)
        y_gt_list_multi  += list(y_gt_multi)
        weight_list_multi += list(y_weights_multi) # could be used for coloring the points

        y_gt_weighted_mean.append(np.sum(np.array(y_gt_multi)*np.array(y_weights_multi))/np.sum(y_weights_multi))
        y_gt_weighted_mean.append(np.sum(np.array(y_gt_multi)*np.array(y_weights_multi))/np.sum(y_weights_multi))

        y_gt_weighted_mean_before_p.append(np.sum(np.array(y_gt_multi_before_p)*np.array(y_weights_multi))/np.sum(y_weights_multi))
        y_gt_weighted_mean_before_p.append(np.sum(np.array(y_gt_multi_before_p)*np.array(y_weights_multi))/np.sum(y_weights_multi))

        y_pred_weighted_mean.append(np.sum(np.array(y_pred)*np.array(y_weights_multi))/np.sum(y_weights_multi))
        y_pred_weighted_mean.append(np.sum(np.array(y_pred)*np.array(y_weights_multi))/np.sum(y_weights_multi))
        x_weighted_mean.append(min_dist)
        x_weighted_mean.append(min_dist+1)


    plt.figure(figsize=(80, 10))

    norm = mcolors.LogNorm(vmin=1, vmax=max(y_weights_multi))
    plt.plot(X_list_single, y_pred_list_single, '--',
            label='fitted curve %s: lmbd=%5.3f, alpha=%5.3f, phi=%5.3f ' % ('powtwo', *tuple(params[0:3])))  
    plt.plot(x_weighted_mean, y_gt_weighted_mean_before_p, alpha=0.5, c='lightblue', label='weighted mean ground truth data - before single p-value')
    plt.plot(x_weighted_mean, y_gt_weighted_mean, c='blue', alpha=0.5, label='weighted mean ground truth data - after single p-value')
    plt.plot(x_weighted_mean, y_pred_weighted_mean, c='darkred', alpha=0.5, label='weighted mean predictions')
    plt.scatter(X_list_multi, y_pred_list_multi, c='darkred', s=10, alpha=0.5, label='multi predictions')
    plt.scatter(X_gt_list_multi, y_gt_list_multi, c=weight_list_multi,  s=10,  label='multi ground truth',cmap='viridis',norm=norm)

    plt.colorbar() # color scale to show the counts for each value

    plt.title(rel_mapping_dict[rule_head] + "(X," + rule_head_const_st + ",T)"  +' <=  '+rel_mapping_dict[rule_body]+"(X," + rule_body_const_st + ",U)", fontsize=10)
    plt.xlabel('time distance')
    plt.ylabel('beta')

    plt.legend()
    # Add secondary ticks TODO
    # ax = plt.gca()  # Get the current axis
    # ax2 = ax.twiny()  # Create a new twin x-axis sharing the same y-axis
    # ax2.set_xticks(min_dist_list)  # Set the locations for secondary ticks
    # ax2.set_xticklabels(min_dist_labels, fontsize=6, rotation=90)
    # ax2.set_xlim(ax.get_xlim())  # Match the x-limits of the primary axis
    plt.grid()
    plt.savefig(figure_path+'/multi_'+str(rule_body)+'-'+str(rule_body_const)+'-'+str(rule_head)+'-'+str(rule_head_const)+'-'+backwardforward+'.png', bbox_inches='tight')
    plt.close('all')


def plot_multi_freq(params, x, y, weights, rul, figure_path, node_mapping_dict, rel_mapping_dict, max_time_window):
    """ plot frequency curve for the new 2025 multi
    """
    # stuff needed for the figure title
    backwardforward = 'F'
    if len(rul) > 2:
        rule_body = rul[2]
        rule_body_const = rul[3]
        rule_head = rul[0]
        rule_head_const = rul[1]
        if rule_head_const in node_mapping_dict:
            rule_head_const_st = node_mapping_dict[rule_head_const][0]
            rule_body_const_st = node_mapping_dict[rule_body_const][0]
        else:
            rule_head_const_st = str(rule_head_const)
            rule_body_const_st = str(rule_body_const)
        if len(rul) > 4:
            backwardforward = 'B'
    else:
        rule_body = rul[0]
        rule_head = rul[1]
        rule_body_const = 'Y'
        rule_head_const = 'Y'
        rule_head_const_st = 'Y'
        rule_body_const_st = 'Y'


    x = np.array(x)
    x1 = x[:,0]
    # x2 = x[1,:]
    sorted_indices = np.argsort(x1)
    x = x[sorted_indices]
    y = np.array(y)[sorted_indices]
    weights = np.array(weights)[sorted_indices]

    y_pred_multi = []
    x_plot = []
    weights_plot = []
    y_plot = []
    counter = 0

    x_dist_of_interest = 1
    x_dist1 = x[:,1]
    indices_dist1 = np.where(x_dist1 == x_dist_of_interest)[0]
    x_d1 = x[indices_dist1] # only take the distances that are 1, i.e. the proportion of elements within the time window
    y_d1 = y[indices_dist1]
    weights_d1 = weights[indices_dist1]
    prop_dict = {}
    for prop, tempdist in x_d1: # sort s.t. we only have one entry per prop
        if prop not in prop_dict:
            prop_dict[prop] = [0,0]
        prop_dict[prop][0] += weights_d1[counter] # sum of weights for this proportion
        prop_dict[prop][1] += y_d1[counter] * weights_d1[counter] # weighted sum of y values for this proportion
        counter +=1

    for prop, (weight_sum, y_weighted_sum) in prop_dict.items():
        x_plot.append(prop)
        weights_plot.append(weight_sum)   
        y_pred_multi.append(score_linear_shift(prop, tempdist, params[3], params[4], params[5]))
        y_plot.append(y_weighted_sum/ weight_sum) # weighted mean of y values for this proportion
        counter += 1
    # plt.figure(figsize=(30, 10))

    plt.figure()


    norm = mcolors.LogNorm(vmin=1, vmax=max(weights))
    plt.scatter(x_plot,y_plot, c=weights_plot, label='ground truth',cmap='viridis',norm=norm)

    plt.plot(x_plot, y_pred_multi, '--',
        # label='Prediction (rho=%5.2f, kappa=%5.2f, gamma=%2.3f)' % (params[3], params[4], params[5]))  
        label = fr'prediction ($\rho$={params[3]:.2f}, $\kappa$={params[4]:.2f}, $\gamma$={params[5]:.2f})')
        #
    # plt.plot(x,y_before_p,  'o', alpha=0.5, color='forestgreen',  label='ground truth before p-value', markerfacecolor='none')    
    plt.colorbar() # color scale to show the counts for each value

    plt.legend()
    plt.ylabel('confidence difference')
    plt.xlabel(fr'relative frequency within time window for $\min(\Delta)=1$')
    plt.title(rel_mapping_dict[rule_head] + "(X," + rule_head_const_st + ",T)"  +' <=  '+rel_mapping_dict[rule_body]+"(X," + rule_body_const_st + ",U)", fontsize=10)
    # Add secondary ticks TODO
    # ax = plt.gca()  # Get the current axis
    # ax2 = ax.twiny()  # Create a new twin x-axis sharing the same y-axis
    # ax2.set_xticks(min_dist_list)  # Set the locations for secondary ticks
    # ax2.set_xticklabels(min_dist_labels, fontsize=6, rotation=90)
    # ax2.set_xlim(ax.get_xlim())  # Match the x-limits of the primary axis
    plt.grid()
    plt.savefig(figure_path+'/multi_frequency'+str(rule_body)+'-'+str(rule_body_const)+'-'+str(rule_head)+'-'+str(rule_head_const)+'-'+backwardforward+'.png', bbox_inches='tight')
    plt.savefig(figure_path+'/pdf_multi_frequency'+str(rule_body)+'-'+str(rule_body_const)+'-'+str(rule_head)+'-'+str(rule_head_const)+'-'+backwardforward+'.pdf', bbox_inches='tight')
    
    plt.close('all')

def plot_curves(rul, x_gt, y_gt, popt,  fct, counts, min_error, fct_type, node_mapping_dict, rel_mapping_dict, multi_flag, figure_path, learn=True, original_betas=[], original_dists=[]):
    """ plot the ground truth data and the fitted curve for a given rule and save to file
    :param x_gt: np.array, the x values of the ground truth data, i.e. delta_t. one entry per value of delta_t,  unique values, not the repeated ones
    :param y_gt: np.array, the y values of the ground truth data. mean entry if multiple entries for the same delta_t,  unique values, not the repeated ones
    :param popt: np.array, the optimal parameters for the fitted curve
    :param rul: tuple, the rule for which the curve is fitted, where rul[0] is the body and rul[1] the head relation
    :param fct: function, the function that was used to fit the curve
    :param counts: np.array, the number of times each x value occurs
    """
    backwardforward = 'F'
    if len(rul) > 2:
        rule_body = rul[2]
        rule_body_const = rul[3]
        rule_head = rul[0]
        rule_head_const = rul[1]
        if rule_head_const in node_mapping_dict:
            rule_head_const_st = node_mapping_dict[rule_head_const][0]
            rule_body_const_st = node_mapping_dict[rule_body_const][0]
        else:
            rule_head_const_st = str(rule_head_const)
            rule_body_const_st = str(rule_body_const)
        if len(rul) > 4:
            backwardforward = 'B'
    else:
        rule_body = rul[0]
        rule_head = rul[1]
        rule_body_const = 'Y'
        rule_head_const = 'Y'
        rule_head_const_st = 'Y'
        rule_body_const_st = 'Y'

    # Use LogNorm to enhance color differentiation for lower values
    norm = mcolors.LogNorm(vmin=1, vmax=max(counts))
 
    y_pred = np.zeros(len(x_gt))
    for i, x in enumerate(x_gt):
        y_pred[i] = fct(x, *popt[:3])
    # y_pred = fct(x_gt, popt[0], popt[1], popt[2])  #  compute the fitted curve. only do this for unique values, not the repeated ones
    plt.figure()
    # plt.plot(x_gt, y_gt, label='ground truth data') 
    # plt.scatter(x_gt, y_gt, c=counts, s=10, cmap='viridis',norm=norm, label='ground truth') # to color code how often each value occurs        
    plt.scatter(x_gt, y_gt, c=counts, cmap='viridis',norm=norm, label='ground truth') # to color code how often each value occurs        
    plt.plot(x_gt, y_pred, '--',
        label = fr'prediction ($\lambda$={popt[0]:.2f}, $\alpha$={popt[1]:.2f}, $\phi$={popt[2]:.2f})')
        # label=f'prediction ($\lambda$=%5.2f, $\alpha$=%5.2f, $\phi$=%5.2f)' % tuple(popt[0:3]))  
    # plt.plot(original_dists, original_betas, 'o', alpha=0.5, label='original data', markerfacecolor='none')     
    plt.colorbar() # color scale to show the counts for each value
    if rule_head in rel_mapping_dict and rule_body in rel_mapping_dict:
        rule_head_str = rel_mapping_dict[rule_head]
        rule_body_str = rel_mapping_dict[rule_body]
    else:
        rule_head_str = str(rule_head)
        rule_body_str = str(rule_body)
    # titlestring = backwardforward+ ' ' +rule_head_str + "(X," + rule_head_const_st + ",T)"  +' <=  '+ rule_body_str+"(X," + rule_body_const_st + ",U)"
    titlestring = rule_head_str + "(X," + rule_head_const_st + ",T)"  +' <=  '+ rule_body_str+"(X," + rule_body_const_st + ",U)"
    if backwardforward == 'F':
        atom = ' & ex. Z with '+ rule_head_str+ '(X,Z,T)'
    else:
        atom = ' & ex. Z with '+ rule_head_str+ '(Z,'+ rule_head_const_st+ ',T)'
    # titlestring += atom % needs to much space
    plt.title(titlestring, fontsize=10)

    plt.xlabel(fr'$\min(\Delta)$')
    plt.ylabel('confidence')
    plt.grid()
    plt.legend()
    if learn == True:
        plt.savefig(figure_path+'/pdf_curvefitted_'+str(rule_body)+'-'+str(rule_body_const)+'-'+str(rule_head)+'-'+str(rule_head_const)+'-'+backwardforward+'.pdf', bbox_inches='tight')
        plt.savefig(figure_path+'/curvefitted_'+str(rule_body)+'-'+str(rule_body_const)+'-'+str(rule_head)+'-'+str(rule_head_const)+'-'+backwardforward+'.png', bbox_inches='tight')
    else:
        plt.savefig(figure_path+'/fixcurve_'+str(rule_body)+'-'+str(rule_body_const)+'-'+str(rule_head)+'-'+str(rule_head_const)+'-'+backwardforward+'.png', bbox_inches='tight')
    plt.close('all')


