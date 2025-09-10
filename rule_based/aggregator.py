import numpy as np
import math 
from itertools import islice

"""
maxplus scores of a ranking will be of the highest predicting rule
candidate discrimination is based on comparing the sequences of predicting
rules confidences lexicographically; note that the outputed scores
are only the ones of the rule with the highest confidence 
noisyor scores and ranking is based on sorting the noisy-or product
the noisy-or sorting is based on -\sum_i(log(1-conf_i)) and transformed
before outputted; this mitigates floating point considerations 
"""

def scoreMax(cand_dict):
    """
    This function ranks the candidates using the max aggregation function, i.e. the candidates
    will be ranked by the rule with the highest score that predicted them.
    """
    def keyfunc(item): 
        return item[1][0]
    
    
    candidate_max_score = {}
    for key, value in cand_dict.items():
        max_tuple = max(value, key=lambda x: x[0])        
        candidate_max_score[key] = (max_tuple[0], [max_tuple[1]]) # Transform the second element into a list

    sorted_candidate_max_score = dict(sorted(candidate_max_score.items(), key=keyfunc, reverse=True))
    
    return sorted_candidate_max_score

def scoreMaxCard(cand_dict):
    """
    This function ranks the candidates using the max card aggregation function, i.e. the candidates
    will be ranked by the rule with the highest score that predicted them. If there is any equality between the
    candidates, there will be a second ranking by the number of the rules that predicted them.
    """
    
    # for each node: find the tuple with the highest score (first element) and the number of rules that predicted the node (second element)
    candidate_max_score = {key: (max(value), len(value)) for key, value in cand_dict.items()}
    
    # sort the dictionary by first: the score and second: the number of rules that predicted the node
    sorted_candidate_max_score = dict(sorted(candidate_max_score.items(), key=lambda item : (item[1][0][0],item[1][1]), reverse=True))

    # transform the sorted dict: key = node, values (score, [rule_identifier])
    sorted_candidate_max_score = {key: (value[0][0], [value[0][1]]) for key, value in sorted_candidate_max_score.items()}

    return sorted_candidate_max_score

def scoreMaxPlus(cand_dict):
    """
    This function ranks the candidates using the maxplus aggregation function, i.e. the candidates
    will be ranked by the rule with the highest score that predicted them. If there is any equality between the
    candidates, there will be a second ranking by the second highest score and so on until a difference can be seen.
    """

    def keyfunc(tup_list): 
        # input: list of tuples [(0.4, 1), (0.3, 2), (0.2, 3)]
        # returns: list with only the first element of each tuple [0.4, 0.3, 0.2]
        entries = []
        for tup in tup_list:
            entries.append(tup[0])
        return entries

    for _, value in cand_dict.items():
        value.sort(key=lambda x: x[0], reverse=True)  # sort for each node the candidates based on their score
    sorted_candidate_maxplus_score = dict(sorted(cand_dict.items(), key=lambda x:keyfunc(x[1]), reverse=True))

    sorted_candidate_maxplus_score = {key: (value[0][0], [value[0][1]]) for key, value in sorted_candidate_maxplus_score.items()}
    
    return sorted_candidate_maxplus_score


def scoreNoisyOr(cand_dict):
    """
    This function ranks the candidates using the noisyor aggregation function, 
    score = 1 - (1-score(r_1)) * ... * (1-score(r_n)), where n = the number of the rules that predicted the candidate
    """
    candidate_noisyor_score = {}
    for key, value in cand_dict.items():
        candidate_noisyor_score[key] = 0 #1 # initial value
        s = 0
        rule_id_keys = []
        for v in value:
            # candidate_noisyor_score[key] *= 1 - v
            if v ==1 :
                s -=-184.20680743952366 #np.log(1e-80)  np.log(1 - v + 1e-80) # using log instead, to avoid problems with small numbers and product going to zero
            else:
                s -= np.log(1 - v[0]) 
            rule_id_keys.append(v[1])
        candidate_noisyor_score[key] = (s, rule_id_keys)
        # candidate_noisyor_score[key] = 1 - candidate_noisyor_score[key]
    sorted_candidate_noisyor_score =  dict(sorted(candidate_noisyor_score.items(), key=lambda item : (item[1][0]), reverse=True)) 

    return sorted_candidate_noisyor_score
def scoreNoisyOrToph(cand_dict, num_top_rules, decay):
    """
    This function ranks the candidates using the noisyor top h aggregation function, 
    Applies a decay factor to the scores of the rules, i.e.
    1. Sorts the rules for each candidate by their score in descending order
    2. Takes the top num_top_rules rules and applies the decay factor to their scores, the second score is multiplied by decay, the third by decay^2 and so on
    score = 1 - (1-score(r_1)) * ... * (1-score(r_h)), where score(r_1) the highest score and so on
    """
    candidate_noisyortoph_score = {}
    for key, value in cand_dict.items():
        candidate_noisyortoph_score[key] = 1 # initial value
        value.sort(reverse=True, key=lambda x: x[0])
        s = 1
        
        rule_id_keys = []
        for i, v in enumerate(value):
            if i >= num_top_rules: break
            xc = v[0] * math.pow(decay, i)
            s *= 1 - xc
            rule_id_keys.append(v[1])

        candidate_noisyortoph_score[key] = (1 - s, rule_id_keys)
    sorted_candidate_noisyortoph_score = dict(sorted(candidate_noisyortoph_score.items(), key=lambda item : item[1][0], reverse=True))
    return sorted_candidate_noisyortoph_score


def aggregation_score(cand_dict, aggregation_func, num_top_rules, decay=1, large_data_flag=False):
    """
    This function calls the the score function specified in aggregation_func
    """
    obj_dict = {}

    if aggregation_func == 'max':
        obj_dict = scoreMax(cand_dict)
    if aggregation_func == 'maxcard':
        obj_dict = scoreMaxCard(cand_dict)
    if aggregation_func == 'maxplus':
        obj_dict = scoreMaxPlus(cand_dict)
    if aggregation_func == 'noisyor' and num_top_rules == -1:
        obj_dict =  scoreNoisyOr(cand_dict) 
    if aggregation_func == 'noisyor' and num_top_rules != -1:
        obj_dict =  scoreNoisyOrToph(cand_dict, num_top_rules, decay)

    
    if large_data_flag:
        # in this case we only return the 200 highest scores
        obj_dict = dict(islice(obj_dict.items(), 200))
        
    return obj_dict
