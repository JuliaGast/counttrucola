from tqdm import tqdm
import numpy as np
import ray
import time
import gc
options = None

import psutil
process = psutil.Process()

from rules import RuleSet
import rule_utils
from aggregator import aggregation_score
from apply_dataset import ApplyDataset


def setOptions(o):
    """
    This function store an options object in a gloabl variable, which 
    allows to use the parameters of the options object without the need
    to pass them as parameters of a function call.
    :param o: a dictionary with the parameters of an option object. 
    """
    global options
    options = o


def apply(rule_dataset, path_rules, path_rankings, progressbar_percentage, evaluation_mode='test', explain_flag=False, options_explain={}, all_rule_types_false=False):
    """
    This function applies the rules with the learned parameter on the test set and writes the rankings to a file

    :return
    """
    # read rules and params from file
    ruleset  = RuleSet(rule_dataset.rels_id_to_string, rule_dataset.nodes_id_to_string)
    print(f'read rules and params from file {path_rules}')

    rules_xy,rules_c,rules_c_backward,num_rules = ruleset.read_rules(path_rules, all_rule_types_false=all_rule_types_false)
    print(f"read {num_rules} rules from file {path_rules}")
    # if we run apply from the explainer: use the options as passed in options_explain
    if explain_flag:
        global options
        options = options_explain

    # all sorts of params that we need
    window_size= options["APPLY_WINDOW_SIZE"]
    aggregation_func, num_top_rules= options["AGGREGATION_FUNCTION"], options["NUM_TOP_RULES"]
    rule_type_z_flag, rule_type_f_flag=options["RULE_TYPE_Z"], options["RULE_TYPE_F"]

    THRESHOLD_APPLY= options["THRESHOLD_APPLY_CONFIDENCE"]
    decay = options["AGGREGATION_DECAY"]
    window_size_freq = options["LEARN_WINDOW_SIZE"]
    num_cpus = options["NUM_CPUS"]
    large_data_flag = rule_dataset.large_data_flag # decides how many prediction scores to store per query
    very_large_data_flag = rule_dataset.very_large_data_flag # for f-rules update per timestep
    if num_cpus > 1:
        print("MEM before ray init" +  str(process.memory_info().rss//1000000))
        use_ray_flag = True
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, log_to_driver=True)
        print(ray.available_resources())
        # ray.timeline(filename="ray_timeline.json")
    else:
        print("MEM at beginning of apply " +  str(process.memory_info().rss//1000000))
        use_ray_flag = False
        

    # create a special data structure for the backward c-rule application direction
    rule_c_by_rhch = {} # contains: key: relh, c_head, value: list of (relb, c_body)
    for relh in rules_c_backward: 
        for (relb, c_head, c_body) in rules_c_backward[relh]:
            k = (relh, c_head)
            if not k in rule_c_by_rhch:
                rule_c_by_rhch[k] = []
            rule_c_by_rhch[k].append((relb, c_body))

    file = open(path_rankings, "w")

    # what data should we predict for; i.e. test queries
    rule_triple_dict = {}
    rule_predicted_dict = {}    

    if explain_flag:
        num_quads = len(rule_dataset.explain_data)
        apply_split = 'explain'
    else:
        apply_split = evaluation_mode
        if evaluation_mode == 'test':
            num_quads = len(rule_dataset.test_data)
        elif evaluation_mode == 'val':
            num_quads = len(rule_dataset.val_data)


    dataset = ApplyDataset(rule_dataset.all_head_rel_tail_t, rule_dataset.all_head_tail_rel_ts, rule_dataset.head_rel_ts[apply_split],
                           num_quads, rule_dataset.inverse_rel_dict)
        
    apply_dict_head_rel_ts = dataset.head_rel_ts # this is the index that we use for the apply function. it is a dictionary with the head as key and a dictionary with rels as keys and timestamps as values
  
    total_iterations = len(apply_dict_head_rel_ts)
    
    # --- starting z-rule data acquisition
    rel2obj2confidence = {} # required for z-rules later    
    if rule_type_z_flag:
        rel2obj2confidence = get_z_rule_stats(rule_dataset, evaluation_mode) 
    # --- end of z-rule data acquisition

    # --- starting f-rule data acquisition
    relsubobj2confidence = {} # required for f-rules later
    if rule_type_f_flag:
        relsubobj2confidence = get_f_rule_stats(rule_dataset, evaluation_mode, very_large_data_flag) 
    # --- end of f-rule data acquisition

    print(f"apply rules to the {evaluation_mode} set")
    # for head in tqdm(testset):
    increment = int(total_iterations*progressbar_percentage) if int(total_iterations*progressbar_percentage) >=1 else 1
    remaining = total_iterations

    if use_ray_flag:
        print("MEM before ray starts " +  str(process.memory_info().rss//1000000))
        apply_ray(apply_dict_head_rel_ts, rule_predicted_dict, relsubobj2confidence, rel2obj2confidence, rule_triple_dict, rules_xy,  
              dataset, rules_c, rule_c_by_rhch, rules_c_backward, window_size, explain_flag, THRESHOLD_APPLY,
              aggregation_func, num_top_rules, decay, window_size_freq,  file, large_data_flag, very_large_data_flag)

    else:
        with tqdm(total=total_iterations) as pbar:
            counter = 0
            for head in apply_dict_head_rel_ts:
                # Update progress bar
                counter += 1
                if counter % increment == 0:
                    remaining -= increment
                    pbar.update(increment)
                if remaining < increment:
                    pbar.update(remaining)
                _= apply_all_rules_head(apply_dict_head_rel_ts[head], head, rule_predicted_dict, relsubobj2confidence, rel2obj2confidence, rules_xy, 
                                                rules_c, rules_c_backward, rule_c_by_rhch,  window_size, 
                                                explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, 
                                                window_size_freq, aggregation_func, num_top_rules, 
                                                decay, file, multi_flag=options["MULTI_FLAG"], large_data_flag=large_data_flag, very_large_data_flag=very_large_data_flag)

    file.close()   

    del apply_dict_head_rel_ts, rule_triple_dict, rel2obj2confidence, relsubobj2confidence
    gc.collect()

    print("MEM at end of apply after deleting stuff " +  str(process.memory_info().rss//1000000))
    
    if explain_flag:
        return num_rules,rule_predicted_dict # rule_triple_dict
    else:
        return num_rules


def apply_ray(apply_dict_head_rel_ts, rule_predicted_dict, relsubobj2confidence, rel2obj2confidence, rule_triple_dict, rules_xy,  
              dataset, rules_c, rule_c_by_rhch, rules_c_backward, window_size, explain_flag, 
              THRESHOLD_APPLY, aggregation_func, num_top_rules, decay, window_size_freq, file, 
              large_data_flag, very_large_data_flag):
    """ do the apply, but with ray parallelization; for this put the test quries in equally sized buckets
    """

    quads_per_task = min((dataset.num_quads // (options['NUM_CPUS']*3))+1, 2000)
    print('quads_per_task: ', quads_per_task)

    # fill the test quadruples in equally sized buckets
    bucket_id = 0
    testset_buckets ={}
    quads_per_bucket_counter = 0
    for head in apply_dict_head_rel_ts:
        for relh in apply_dict_head_rel_ts[head]:
            for t in apply_dict_head_rel_ts[head][relh]:                                      
                query = (head, relh, t)
                if quads_per_bucket_counter >= quads_per_task:
                    bucket_id += 1
                    quads_per_bucket_counter = 0
                if bucket_id not in testset_buckets:
                    testset_buckets[bucket_id] = {}
                if head not in testset_buckets[bucket_id]:
                    testset_buckets[bucket_id][head] = {}
                if relh not in testset_buckets[bucket_id][head]:
                    testset_buckets[bucket_id][head][relh] = []

                testset_buckets[bucket_id][head][relh].append(t)
                quads_per_bucket_counter +=1

    relsubobj2confidence_buckets = {}
    if relsubobj2confidence:
        for bucket in testset_buckets:
            relsubobj2confidence_buckets[bucket] = {}
            for head in testset_buckets[bucket]:
                for relh in testset_buckets[bucket][head]:
                    if relh in relsubobj2confidence:
                        if head in relsubobj2confidence[relh]:
                            if relh not in relsubobj2confidence_buckets[bucket]:
                                relsubobj2confidence_buckets[bucket][relh] = {}
                            relsubobj2confidence_buckets[bucket][relh][head] = relsubobj2confidence[relh][head]

    del relsubobj2confidence

    dataset_id = ray.put(dataset)
    rule_predicted_dict_id = ray.put(rule_predicted_dict)
    # relsubobj2confidence_id = ray.put(relsubobj2confidence)
    rel2obj2confidence_id = ray.put(rel2obj2confidence)
    rule_triple_dict_id = ray.put(rule_triple_dict)
    rules_xy_id = ray.put(rules_xy)
    rules_c_id = ray.put(rules_c)
    rule_c_by_rhch_id = ray.put(rule_c_by_rhch)
    rules_c_backward_id = ray.put(rules_c_backward)
    print("MEM after ray.put " +  str(process.memory_info().rss//1000000))
    # print('collecting all tasks for ray')
    tasks = []
    for task_bucket in testset_buckets:
        quads_bucket = testset_buckets[task_bucket]
        
        quads_bucket_id = ray.put(quads_bucket)
        relsubobj2confidence_bucket = relsubobj2confidence_buckets[task_bucket] if relsubobj2confidence_buckets else {}
        relsubobj2confidence_id = ray.put(relsubobj2confidence_bucket) if relsubobj2confidence_bucket else ray.put({})
        tasks.append(apply_all_rules_ray.remote(quads_bucket_id, rule_predicted_dict_id, relsubobj2confidence_id, rel2obj2confidence_id, rules_xy_id, 
                                            rules_c_id, rules_c_backward_id, rule_c_by_rhch_id, window_size, 
                                            explain_flag, THRESHOLD_APPLY, rule_triple_dict_id, dataset_id, window_size_freq,
                                            aggregation_func, num_top_rules, decay, options['MULTI_FLAG'], large_data_flag, very_large_data_flag))

    print("MEM after appending all ray tasks " +  str(process.memory_info().rss//1000000))
    print('starting the parallel ray tasks ')
    with tqdm(total=len(tasks)) as pbar:
        pending = tasks
        while pending:

            done, pending = ray.wait(pending, num_returns=min(options['NUM_CPUS'], len(pending)))
            while done:
                task = done.pop(0)
                # result_dict_head_id = ray.get(task)
                result_dict_head = ray.get(task)

                for query in result_dict_head:
                    head, relh, t = query
                    file.write(str(head) + " " + str(relh) + " " + "?" + " " + str(t) + "\n")
                    sorted_obj_dict = result_dict_head[query]
                    for i, o in enumerate(sorted_obj_dict):
                        if i < len(sorted_obj_dict) - 1: file.write(str(o) + " " + str(sorted_obj_dict[o][0]) + " ")
                        else: file.write(str(o) + " " + str(sorted_obj_dict[o][0]))    
                    file.write("\n")  
                pbar.update(1)
    print("finished all tasks, now closing the file")          
    file.close()    
    print("MEM after finising all ray tasks " +  str(process.memory_info().rss//1000000))

@ray.remote(num_cpus=1)
def apply_all_rules_ray(testset_buckets, rule_predicted_dict, relsubobj2confidence, rel2obj2confidence, rules_xy, 
                                            rules_c, rules_c_backward, rule_c_by_rhch, window_size, 
                                            explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq,
                                            aggregation_func, num_top_rules, decay, multi_flag, large_data_flag, very_large_data_flag):
    result_dict = {}
    time.sleep(0.1)

    for head in testset_buckets:
        result_dict_head = apply_all_rules_head(testset_buckets[head], head, rule_predicted_dict, relsubobj2confidence, rel2obj2confidence, rules_xy, 
                                                 rules_c, rules_c_backward, rule_c_by_rhch,  window_size, 
                                                 explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq,
                                                 aggregation_func, num_top_rules, decay, file=None, multi_flag=multi_flag, large_data_flag=large_data_flag, 
                                                 very_large_data_flag=very_large_data_flag)
        result_dict.update(result_dict_head)

    return result_dict


def apply_all_rules_head(testset_head, head, rule_predicted_dict, relsubobj2confidence, rel2obj2confidence, rules_xy, 
                                            rules_c, rules_c_backward, rule_c_by_rhch,  window_size, 
                                            explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq,
                                            aggregation_func, num_top_rules, decay, file, multi_flag, large_data_flag, very_large_data_flag):
    """ apply all rules for a given head.
    """

    result_dict = {}
    for relh in testset_head:

        for t in testset_head[relh]:
            query = (head, relh, t)
            rule_predicted_dict[query] = {}
            rule_counter = 1
            

            if file:
                file.write(str(head) + " " + str(relh) + " " + "?" + " " + str(t) + "\n")
            else:
                result_dict[query] = {}
            obj_dict = apply_all_rules_quad(head, relh, t, query, relsubobj2confidence, rel2obj2confidence, rules_xy, 
                                        rules_c, rules_c_backward, rule_c_by_rhch,  window_size, 
                                        rule_counter, explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq, 
                                         multi_flag, very_large_data_flag)
            
            # --- sort and process all candidates collected and write rankings in file ----
            if obj_dict:
                # section 4.3 Aggregation
                sorted_obj_dict = aggregation_score(obj_dict, aggregation_func, num_top_rules, decay, large_data_flag) # aggregation, e.g. max or noisyor
                
                if file:
                    for i, o in enumerate(sorted_obj_dict):
                        if i < len(sorted_obj_dict) - 1: file.write(str(o) + " " + str(sorted_obj_dict[o][0]) + " ")
                        else: file.write(str(o) + " " + str(sorted_obj_dict[o][0]))      
                else:
                    result_dict[query] = sorted_obj_dict
            if file:
                file.write("\n")      

            # --- explanations --- 
            if obj_dict and explain_flag:
                for node, ids in sorted_obj_dict.items():
                    score = ids[0]
                    for rule_counter_id in ids[1]:
                        rule_of_interest = rule_triple_dict[query][rule_counter_id] # (score, rul, obj, params, quad_firing)
                        if not query in rule_predicted_dict:
                            rule_predicted_dict[query] = {}
                        if not node in rule_predicted_dict[query]:
                            rule_predicted_dict[query][node] = {}
                        if not score in rule_predicted_dict[query][node]:
                            rule_predicted_dict[query][node][score] = []
                        rule_predicted_dict[query][node][score].append(rule_of_interest)
            t6 = time.time()
    return result_dict


def apply_all_rules_quad(head, relh, t, query, relsubobj2confidence, rel2obj2confidence, rules_xy, rules_c, rules_c_backward, rule_c_by_rhch, 
                         window_size,  rule_counter, explain_flag, THRESHOLD_APPLY, rule_triple_dict, 
                         dataset, window_size_freq, multi_flag, very_large_data_flag):
    """ apply all rules for a given head, relh and t.
    """

    # checking if the triple resulting from the body relation with the same groundings in the head relation is in the test set
    obj_dict = {} # i (christian) had to move this one level up ... hope it still works
    # --- apply f-rules
    rule_counter = apply_f_rules(relsubobj2confidence, relh, head, t, obj_dict, rule_counter, explain_flag, query, rule_triple_dict, very_large_data_flag)

    
    # --- apply z-rules --- 
    rule_counter = apply_z_rules(rel2obj2confidence, relh, obj_dict, rule_counter, explain_flag, query, rule_triple_dict)

    # --- apply XY rules ----
    rule_counter = apply_xy_rules(relh, head, t, rules_xy,  window_size, obj_dict, rule_counter, explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq,  multi_flag)

    # --- apply C rules direcly with constant in object position ----
    rule_counter = apply_c_rules_forward(relh, head, t, rules_c, window_size, obj_dict, rule_counter, explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq, multi_flag)
    t3 = time.time()
    
    # --- apply C rules indirectly using a virtual rule with constants in subject position ----
    # what we have before we enter here: head, relh, tail                        
    # For query  invclimbs(l3, ?,T) we need the rule climbs(X,l3,T) <- climbs(X, l2, U) & exists Z with climbs(Z, l3, T)  -> backward-inverse-rule                              
    t4 = time.time()
    rule_counter = apply_c_rules_backward(relh, head,  t, rule_c_by_rhch, rules_c_backward, window_size,  obj_dict,
                rule_counter, explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq, multi_flag)
    
    return obj_dict



def apply_xy_rules(relh, head, t, rules_xy, window_size_apply,  obj_dict, rule_counter, explain_flag, 
                   THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq,  multi_flag):
    all_head_rel_tail_t = dataset.all_head_rel_tail_t
    if relh in rules_xy:
        for relb in rules_xy[relh]: 
            if relb in all_head_rel_tail_t[head]:
                for obj in all_head_rel_tail_t[head][relb]:
                    # list_of_ts list should contain the recent timestamps that are less than t (t is the timestamp of the query)
                    (t_recent, list_of_ts) = filter_timesteps_within_apply(t, all_head_rel_tail_t[head][relb][obj], window_size_apply, window_size_freq, multi_flag=multi_flag)
                    
                    # check if  t_
                    t_recent 
                    
                    params = rules_xy[relh][relb]
                    rulekey = (relh, relb)
                    query_gt = (head, relh,  t) # for logging the explanations - not used for predictions
                    triple_firing = (head, relb, obj) # the cause for the rule prediction

                    add_scored_object(t, t_recent, list_of_ts, obj, obj_dict, params, THRESHOLD_APPLY, multi_flag, 
                                        explain_flag, rulekey, rule_triple_dict, query_gt, triple_firing, rule_counter, window_size_freq=window_size_freq)
                    rule_counter += 2
    return rule_counter

def apply_f_rules(relsubobj2confidence, relh, head, t, obj_dict, rule_counter, explain_flag, query, rule_triple_dict, very_large_data_flag):
    # print(str(t))
    # take care: relsubobj2confidence contains (nominator,denominator) which yields confidence
    if relh in relsubobj2confidence:
        if head in relsubobj2confidence[relh]:
            for obj in relsubobj2confidence[relh][head]:

                firing_trip =  tuple([((relh, relh, obj, head))]) # tuple([(head, relh, obj, -1)])
                (noms,denoms) = relsubobj2confidence[relh][head][obj]
                if very_large_data_flag: # then we did not update the confidence per timestep.
                    if denoms == 0: conf = 0
                    else:
                        # Find the largest key in noms that is <= t
                        keys = [k for k in noms.keys() if k <= t]
                        if keys:
                            nearest_t = max(keys)
                            conf = noms[nearest_t] / denoms[nearest_t] if denoms[nearest_t] != 0 else 0
                        else:
                            conf = 0
                    # else: conf = noms / denoms
                else:
                    if denoms[t] == 0: conf = 0
                    else: conf = noms[t] / denoms[t]

                if conf > 0: # this is also needed to avoid test set leakage

                    if not obj in obj_dict:
                        obj_dict[obj] = []
                    obj_dict[obj].append((conf, rule_counter, firing_trip))
                
                    if explain_flag:
                        rule_key = (relh, head, obj) 
                        quad_firing = (head, relh, -1, -1, -1, [])
                        if query in rule_triple_dict:
                            if rule_counter in rule_triple_dict[query]:
                                print('alarm f')
                            rule_triple_dict[query][rule_counter] = {}
                        else:
                            rule_triple_dict[query] = {}
                        rule_triple_dict[query][rule_counter] = (conf, (rule_key), obj, (), quad_firing, 's', conf, 0) # (score, rul, obj, params, quad_firing)
                        # (score, rul, obj, params, quad_firing, 'm', score_single, score_multi)
                    rule_counter+=1
    return rule_counter


def apply_z_rules(rel2obj2confidence, relh, obj_dict, rule_counter, explain_flag, query, rule_triple_dict):
    if relh in rel2obj2confidence:
        for zobj in rel2obj2confidence[relh]:
            if not zobj in obj_dict:
                obj_dict[zobj] = []
            firing_trip = tuple([(relh, zobj)]) #head_rel, body_rel and body_ob
            

            eq_rules = firing_trip

            obj_dict[zobj].append((rel2obj2confidence[relh][zobj], rule_counter, eq_rules))

            # explanation
            if explain_flag:
                quad_firing = (-1, relh, zobj, -1, -1, [])
                if query in rule_triple_dict:
                    if rule_counter in rule_triple_dict[query]:
                        print('alarm z')
                    rule_triple_dict[query][rule_counter] = {}
                else:
                    rule_triple_dict[query] = {}
                rule_triple_dict[query][rule_counter] = (rel2obj2confidence[relh][zobj], (), zobj, (), quad_firing, 's', rel2obj2confidence[relh][zobj], 0) 
            rule_counter+=1
    return rule_counter
# helper function called from within the apply function
# adds an candidate object with its score to the dictionary that stores scored candidates

def apply_c_rules_forward(relh, head, t, rules_c, window_size_apply, obj_dict, rule_counter, explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq, multi_flag):
    if relh in rules_c:
        for (relb, c_head, c_body) in rules_c[relh]:
            list_of_ts_all = dataset.get_t_when_true_all(head, relb, c_body)
            (t_recent, list_of_ts) = filter_timesteps_within_apply(t, list_of_ts_all, window_size_apply, window_size_freq, multi_flag=multi_flag)
            obj = c_head
            params = rules_c[relh][(relb, c_head, c_body)]
            rulekey = (relh,  c_head, relb, c_body)
            query_gt = (head, relh,  t) # for logging the explanations - not used for predictions
            triple_firing = (head, relb, c_body)

            add_scored_object(t, t_recent, list_of_ts, obj, obj_dict, params, THRESHOLD_APPLY, multi_flag, 
                                explain_flag, rulekey, rule_triple_dict, query_gt, triple_firing, rule_counter,  
                                window_size_freq=window_size_freq)
            rule_counter += 2
    return rule_counter
    
def apply_c_rules_backward(relh, head,  t, rule_c_by_rhch, rules_c_backward, window_size_apply,  obj_dict,
        rule_counter, explain_flag, THRESHOLD_APPLY, rule_triple_dict, dataset, window_size_freq, multi_flag):
    relh_inv = dataset.inverse_rel_dict[relh]
    if (relh_inv, head) in rule_c_by_rhch: 
        for (relb, c_body) in rule_c_by_rhch[(relh_inv, head)]:
            heads2ts = dataset.get_heads_all(relb, c_body) # all monkeys that have dropped to level2 before and their timesteps
            recurrency_c_rule = (relh_inv == relb and head == c_body)
            for x in heads2ts: #x is 
                (t_recent, list_of_ts) = filter_timesteps_within_apply(t, heads2ts[x], window_size_apply, window_size_freq, multi_flag=multi_flag)
                params = rules_c_backward[relh_inv][(relb, head, c_body)] # ??? maybe better ignore question marks
                

                rulekey = (relh_inv, head, relb, c_body, 'b')  # (relh,  c_head, relb, c_body) 
                query = (head, relh,  t)
                relb_inv = dataset.inverse_rel_dict[relb]

                triple_firing = (c_body, int(relb_inv), x) # e.g. for inv_mkestatement(x, mohammad, T) <- inv_call(x, mohammad T) we need (mohammad, call, x) as firing triple



                add_scored_object(t, t_recent, list_of_ts, x, obj_dict, params, THRESHOLD_APPLY, multi_flag, explain_flag, 
                                  rulekey, rule_triple_dict, query, triple_firing, rule_counter, 
                                    window_size_freq=window_size_freq)
                rule_counter+=2

def add_scored_object(t, t_recent, list_of_ts, obj, obj_dict, params, threshold, multi_flag, explain_flag=False, rul=None, rule_triple_dict={}, query=None, 
                      triple_firing=None, rule_counter=0, window_size_freq=50):
    if t_recent > 0:
        # --- compute score ---
        score = 0.0
        delta_t = t - t_recent
        

        # compute single score
        score_single = rule_utils.score_single(delta_t, *params[0:3])
        
        # print("delta_t=" + str(delta_t) + " score=" + str(score_single))

        # compute multi deviation score
        score_multi = 0
        if multi_flag:
            # min_t = t - window_size_freq if t - window_size_freq > 0 else 0
            # list_of_ts2 = list(filter(lambda x : x >= min_t, list_of_ts)) # occurence timestep within the last window_size_freq
            freq = (len(list_of_ts))/window_size_freq  
            # score_multi =  float(rule_utils.score_linear(freq, *params[3:6]))
            score_multi =  float(rule_utils.score_linear_shift(freq, delta_t, *params[3:6]))
            # print(str(len(list_of_ts)) + ": " + str(score_multi))

        fade_factor= rule_utils.fade_away_factor(delta_t, window_size_freq)
        score_multi = fade_factor*(score_multi) # if the score is farther away, then fade it out 

        # the score is the sum of the single score and the multi deviation score        
        score = score_single + score_multi    
        score = np.clip(score, 0, 1) # clip the score to  be between 0 and 1

        if score > threshold: # this is THRESHOLD_APPLY_CONFIDENCE
            if obj not in obj_dict: obj_dict[obj] = []

            # ---  nonred noisyor: find redundant rules --- 
            # this contains the (head_rel, body_rel and body_obj, body_subj) for the firing rule as well as (head_rel, body_rel_t and body_obj, body_subj) for all body_rels_i whhere rels_i are equivalent to body_rel
            eq_rules = tuple()

            # ---  append the score, the rule_counter and the equivalent rules to the obj_dict ---
            obj_dict[obj].append((score, rule_counter, eq_rules))  # last entry is ((head_rel, body_rel and body_obj and body_sub)) with all equivalent rules

            # ---  explanations ---
            if explain_flag:
                explanation_prep(rule_triple_dict, triple_firing, list_of_ts, t_recent, query, rule_counter, score, rul, obj, params, score_single, score_multi)


def explanation_prep(rule_triple_dict, triple_firing, list_of_ts, t_recent, query, rule_counter, score, rul, obj, params, score_single, score_multi):
    """ put all the infos needed for explanation for the prediction of the specific quad and rule in rule_triple_dict
    """
    # in the following line i (christian) added list_of_ts != None to avoid null pointer exception
    if list_of_ts != None and len(list_of_ts) > 0:
        quad_firing = (triple_firing[0], triple_firing[1], triple_firing[2], max(list_of_ts), len(list_of_ts))
    else:
        quad_firing = (triple_firing[0], triple_firing[1], triple_firing[2], t_recent, 0)
    if query in rule_triple_dict:
        if rule_counter in rule_triple_dict[query]:
            print('alarm explain')
        rule_triple_dict[query][rule_counter] = {}
    else:
        rule_triple_dict[query] = {}
    rule_triple_dict[query][rule_counter] = (score, rul, obj, params, quad_firing, 'm', score_single, score_multi)


# helper function called from within apply
def filter_timesteps_within_apply(t, list_of_ts_all, window_size_apply, window_size_freq, multi_flag):
    if multi_flag:
        return filter_timesteps_within_apply_multi(t, list_of_ts_all, window_size_apply, window_size_freq)
    else:
        return filter_timesteps_within_apply_single(t, list_of_ts_all, window_size_apply)
    

def filter_timesteps_within_apply_multi(t, list_of_ts_all, window_size_apply, window_size_freq):
    ts_recent = -1
    list_of_ts = [] 
    for tt in reversed(list_of_ts_all):
        if tt < t:
            distance = t - tt
            if distance > window_size_freq and ts_recent > 0: break
            if window_size_apply > 0 and distance > window_size_apply: break
            if ts_recent < 0: ts_recent = tt
            if distance < window_size_freq:
                list_of_ts.append(tt)
    list_of_ts.reverse()
    return (ts_recent, list_of_ts)
    
# helper function called from within apply
def filter_timesteps_within_apply_single(t, list_of_ts_all, window_size_apply):
    ts = -1
    for tt in reversed(list_of_ts_all):
        if tt < t:
            ts = tt
            if window_size_apply > 0 and t - tt > window_size_apply:
                ts = -1 # reset to -1 if outside the apply window
            break
    return (ts, None)

def get_z_rule_stats(dataset, evaluation_mode):
    print("gathering z-rule statistics ...")
    Z_MIN_SUPPORT = options["Z_MIN_SUPPORT"]
    Z_MIN_CONFIDENCE = options["Z_MIN_CONFIDENCE"]
    Z_UNSEEN_NEGATIVES = options["Z_UNSEEN_NEGATIVES"]
    Z_RULES_FACTOR = options["Z_RULES_FACTOR"]
    rel2count = {}
    rel2obj2count = {}
    rel2obj2confidence = {}
    # collect examples counts from train
    head_tail_rel_t = dataset.head_tail_rel_t["train"]
    for h in head_tail_rel_t:
        for t in head_tail_rel_t[h]:
            for r in head_tail_rel_t[h][t]:
                num_of_ts = len(head_tail_rel_t[h][t][r])
                #print(" => " + str(num_of_ts))
                if r not in rel2count: rel2count[r] = 0
                rel2count[r] = rel2count[r] + num_of_ts
                if r not in rel2obj2count: rel2obj2count[r] = {}
                if t not in rel2obj2count[r]: rel2obj2count[r][t] = 0
                rel2obj2count[r][t] = rel2obj2count[r][t] + num_of_ts
    if evaluation_mode == 'test':
    # collect examples counts from valid
        head_tail_rel_t = dataset.head_tail_rel_t["val"]
        for h in head_tail_rel_t:
            for t in head_tail_rel_t[h]:
                for r in head_tail_rel_t[h][t]:
                    num_of_ts = len(head_tail_rel_t[h][t][r])
                    #print(" => " + str(num_of_ts))
                    if r not in rel2count: rel2count[r] = 0
                    rel2count[r] = rel2count[r] + num_of_ts
                    if r not in rel2obj2count: rel2obj2count[r] = {}
                    if t not in rel2obj2count[r]: rel2obj2count[r][t] = 0
                    rel2obj2count[r][t] = rel2obj2count[r][t] + num_of_ts
    head_tail_rel_t = dataset.head_tail_rel_t["train"]
    # compute confidence and store the z rules above the threshold
    for r in rel2obj2count:
        for t in rel2obj2count[r]:
            if rel2obj2count[r][t] >= Z_MIN_SUPPORT:
                conf = rel2obj2count[r][t] / (rel2count[r] + Z_UNSEEN_NEGATIVES)
                if conf >= Z_MIN_CONFIDENCE:                
                    if r not in rel2obj2confidence: rel2obj2confidence[r] = {}
                    rel2obj2confidence[r][t] = conf * Z_RULES_FACTOR
    print("... done with gathering z-rule statistics")

    return rel2obj2confidence


def get_f_rule_stats(dataset, evaluation_mode, very_large_data_flag):
    print('gather f-rule statistics ...')
    relsubobj2scores = {}


    F_UNSEEN_NEGATIVES = options["F_UNSEEN_NEGATIVES"]


    min_t_target, max_t_target = dataset.min_timestep[evaluation_mode], dataset.max_timestep[evaluation_mode]

    print()
    print("MEM at beginning of f-rule aquisition: " +  str(process.memory_info().rss//1000000))
    num_f_rules= 0
    with tqdm(total=len(dataset.rel_head_tail_t['train'])) as pbar:
        for rel in dataset.rel_head_tail_t['train']:
            allsubs = set()
            if rel in dataset.rel_head_tail_t['train']: allsubs.update(dataset.rel_head_tail_t['train'][rel].keys())
            if evaluation_mode == 'val':
                if rel in dataset.rel_head_tail_t['val']: allsubs.update(dataset.rel_head_tail_t['val'][rel].keys())
            if evaluation_mode == 'test':
                if rel in dataset.rel_head_tail_t['val']: allsubs.update(dataset.rel_head_tail_t['val'][rel].keys())
                if rel in dataset.rel_head_tail_t['test']: allsubs.update(dataset.rel_head_tail_t['test'][rel].keys())
            # in the code above no leakage, you are wrong of you think so (probably)
            for sub in allsubs:
                num_app = F_UNSEEN_NEGATIVES # number of appearences for rel(sub,?,?)
                num_app_sub_obj = {} # number of appearences for rel(sub,obj,?) where obj is the key of the dictionary
                # fetch everthing from train
                if rel in dataset.rel_head_tail_t['train'] and sub in dataset.rel_head_tail_t['train'][rel]:
                    for obj in dataset.rel_head_tail_t['train'][rel][sub]:
                        sum_t = len(dataset.rel_head_tail_t['train'][rel][sub][obj])
                        num_app += sum_t
                        num_app_sub_obj[obj] = sum_t
                # fetch everthing from valid, if the target is test
                if evaluation_mode == 'test':
                    if rel in dataset.rel_head_tail_t['val'] and sub in dataset.rel_head_tail_t['val'][rel]:
                        for obj in dataset.rel_head_tail_t['val'][rel][sub]:
                            sum_t = len(dataset.rel_head_tail_t['val'][rel][sub][obj])
                            num_app += sum_t
                            if not obj in num_app_sub_obj: num_app_sub_obj[obj] = 0
                            num_app_sub_obj[obj] += sum_t
                # collect all appearances of obj with rel(sub,obj,?) in the target datasets
                if rel in dataset.rel_head_tail_t[evaluation_mode] and sub in dataset.rel_head_tail_t[evaluation_mode][rel]:
                    for obj in dataset.rel_head_tail_t[evaluation_mode][rel][sub]:
                        if not obj in num_app_sub_obj:
                            num_app_sub_obj[obj] = 0
                
                if very_large_data_flag: # then do not update per timestep
                    if not rel in relsubobj2scores: relsubobj2scores[rel] = {}
                    if not sub in relsubobj2scores[rel]: relsubobj2scores[rel][sub] = {}
                    relsubobj2scores[rel][sub][obj] = (num_app_sub_obj[obj], num_app)

                    num_app_per_t = {}
                    num_app_per_t[min_t_target] = num_app
                    num_app_sub_obj_per_t = {}
                    for obj in num_app_sub_obj:
                        num_app_sub_obj_per_t[obj] = {}
                        num_app_sub_obj_per_t[obj][min_t_target] = num_app_sub_obj[obj]
                    # now everthing is prepared, the initial values are already stored
                    # do the real work now, count within the part of the dataset identified as  {0, ...., t-1} 
                    
                    step_size = 50 # only update f-rules every step_size timesteps -> more memory efficient
                    t_index_counter = np.arange(min_t_target, max_t_target, step_size) # only update every i-th timestep
                    num_app_sub_obj_group = num_app_sub_obj

                    occ_count = 0
                    for t in range(min_t_target, max_t_target):
                        
                        for obj in num_app_sub_obj:
                            # if t in num_app_sub_obj_per_t[obj]: # only update every 10th timestep
                            if dataset.is_true(sub, rel, obj, t, evaluation_mode):
                                occ_count +=1
                                # print(str(t) + ": min=" + str(min_t_target))
                                num_app_sub_obj_group[obj] = num_app_sub_obj_group[obj] + 1
                                # num_app_sub_obj_per_t[obj][t+1] = num_app_sub_obj_per_t[obj][t]
                            else:
                                num_app_sub_obj_group[obj] = num_app_sub_obj_group[obj]                         

                            if t + 1 in t_index_counter: # only update every 10th timestep
                                num_app_sub_obj_per_t[obj][t+1] = num_app_sub_obj_group[obj]
                                num_app_per_t[t+1] = num_app_per_t[t+1-step_size] + occ_count
                                occ_count = 0


                        # num_app_per_t[t+1] = num_app_per_t[t] + occ_count
                        # num_app_per_t[t+1] = num_app_per_t[t]
                    # and now put it into the datrastruture that is teh return value
                    if not rel in relsubobj2scores: relsubobj2scores[rel] = {}
                    if not sub in relsubobj2scores[rel]: relsubobj2scores[rel][sub] = {}
                    for obj in num_app_sub_obj:
                        relsubobj2scores[rel][sub][obj] = (num_app_sub_obj_per_t[obj], num_app_per_t)
                        num_f_rules += 1
                # now do the real work and count per time point whats going on
                else: # one score per test timestep
                    num_app_per_t = {}
                    num_app_per_t[min_t_target] = num_app
                    num_app_sub_obj_per_t = {}
                    for obj in num_app_sub_obj:
                        num_app_sub_obj_per_t[obj] = {}
                        num_app_sub_obj_per_t[obj][min_t_target] = num_app_sub_obj[obj]
                    # now everthing is prepared, the initial values are already stored
                    # do the real work now, count within the part of the dataset identified as  {0, ...., t-1} 
                    
                    for t in range(min_t_target, max_t_target):
                        occ_count = 0
                        for obj in num_app_sub_obj:
                            if dataset.is_true(sub, rel, obj, t, evaluation_mode):
                                occ_count +=1
                                # print(str(t) + ": min=" + str(min_t_target))
                                num_app_sub_obj_per_t[obj][t+1] = num_app_sub_obj_per_t[obj][t] + 1
                                # num_app_sub_obj_per_t[obj][t+1] = num_app_sub_obj_per_t[obj][t]
                            else:
                                num_app_sub_obj_per_t[obj][t+1] = num_app_sub_obj_per_t[obj][t]
                        num_app_per_t[t+1] = num_app_per_t[t] + occ_count
                        # num_app_per_t[t+1] = num_app_per_t[t]
                    # and now put it into the datrastruture that is teh return value
                    if not rel in relsubobj2scores: relsubobj2scores[rel] = {}
                    if not sub in relsubobj2scores[rel]: relsubobj2scores[rel][sub] = {}
                    for obj in num_app_sub_obj:
                        relsubobj2scores[rel][sub][obj] = (num_app_sub_obj_per_t[obj], num_app_per_t)
                        num_f_rules += 1
            pbar.update(1)
    print(f"... done with gathering f-rule statistics, found {num_f_rules} f-rules.")
    print("MEM after f-rule aquisition: " +  str(process.memory_info().rss//1000000))
    return relsubobj2scores

def read_number_of_rules(dataset, path_rules):
    """
    This function applies the rules with the learned parameter on the test set 
    :param dataset: the needed dataset object for data preprocessing
    :param learn_option: flag, whether the parameters should be learned or should the default or the static ones be used
    :param path_rules: the path of the file where the rules should be loaded from
    :param path_rankings: the path of the file where the rankings should be written to
    :param flag: should specify which rule types need to be used. "all" for all rule types, 
                "rec" for recurrency and "non-rec" for non-recurrency
    :return
    """
    ruleset  = RuleSet(dataset.rels_id_to_string, dataset.nodes_id_to_string)
    print(f'read rules and params from file {path_rules}')
    rules_xy, rules_c, rules_c_backward, num_rules = ruleset.read_rules( path_rules)

    return num_rules
