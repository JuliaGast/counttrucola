import pickle
import timeit 
import pickle
from tqdm import tqdm

import random

import psutil
process = psutil.Process()


def load_or_make_learn_input(options, learn_data_path, ruledataset):
    """ depending on the options, this function either loads the learn input from a file in learn_data_path or creates it
    it also saved the learn input to a file in learn_data_path
    it takes into account whether to get the learn input for single or multi, and whether we need rules with constants or not
    """
    create_learn_data_flag = options["CREATE_LEARN_DATA_FLAG"]
    if create_learn_data_flag ==False:
        print("-----------------do not create the learn data - load it from file")
        try:
            learn_input = load_learn_input(learn_data_path) 
        except:
            print('could not load learn data from file: ', learn_data_path, 'create learn data instead')
            create_learn_data_flag = True

    if create_learn_data_flag:
        print('-----------------create learn data which is used to learn the params for confidence functions')
        if options["MULTI_FLAG"]:
            learn_input = {}
            if options["RULE_TYPE_CYC1_REC"] or options["RULE_TYPE_CYC1_NON_REC"]:
                learn_input = make_learn_input_multi(ruledataset, options["LEARN_WINDOW_SIZE"], 0.01, options['RR_OFFSET'])
            if options["RULE_TYPE_C"]:
                extend_learn_input_constants_multi(ruledataset, learn_input, options, 0.01)
        else: # SINGLE       
            learn_input = {}
            if options["RULE_TYPE_CYC1_REC"] or options["RULE_TYPE_CYC1_NON_REC"]:
                learn_input = make_learn_input(ruledataset, options["LEARN_WINDOW_SIZE"], 0.01, options['RR_OFFSET']) 
            if options["RULE_TYPE_C"]:
                extend_learn_input_constants_single(ruledataset, learn_input, options, 0.01)
        # save examples to file, in both MULTI and SINGLE case
        with open(learn_data_path, "wb") as file: pickle.dump(learn_input, file)

    return learn_input

def load_learn_input(learn_data_path):
    """
    This function loads the learn input from a file
    :param learn_data_path: the path of the file where the learn data should be loaded from
    :return examples: a dictionary with the following structure:
    {(body_relation, head_relation) : { delta_t1 : (true predictions, all predictions),  delta_t2 : (true predictions, all predictions), ... }}
    """
    print(">>> load the examples required as input to learn confidence functions from ", learn_data_path)
    with open(learn_data_path, "rb") as file:
        examples = pickle.load(file)
    return examples


def make_learn_input_multi(dataset, window_size, progressbar_percentage, RR_OFFSET):
    """
    This function generates the learn input for the model taking into account multiple body groundings.
    :param dataset: the needed dataset object for data preprocessing
    :param window_size: the windowsize to check for body groundings
    :param progressbar_percentage: for the increment of the bar progress, set 0 if you want to increase the progress by each triple
    :param RR_OFFSET: The rr offset
    :return examples: a dictionary with the structure 
     {(body_relation, head_relation) : { (3,5,8) : (true predictions, all predictions),  (1,2) : (true predictions, all predictions), ... }}
    The first list contains the positive examples. The second list contains the negative examples.
    Each tuple contrtains the time distances to all body groundings within the time window that made the rule fire.
    """
    # TODO we need to be able to specify what kinds of rules we want here?
    print(">>> starting to collect the examples required to learn confidence functions for xy-rules (multi setting)")
    examples_ff = {} 
    start_beginning_flag = True # If False the first time steps will not be used until the window size is reached, if true all timesteps will be used

    # count different triples which is required for entsimating required / remaining time
    num_of_triples = 0
    for head in dataset.head_tail_rel_t["train"]:
        num_of_triples += len(dataset.head_tail_rel_t["train"][head])
    # TODO CONTINUE HERE

    # collect the examples
    # Initialize tqdm progress bar with the total number of triples
    increment = int(num_of_triples*progressbar_percentage) if int(num_of_triples*progressbar_percentage) >=1 else 1
    remaining = num_of_triples
        
    with tqdm(total=num_of_triples, desc=">>> collecting examples for xy-rules:", unit="triple") as pbar:
        counter = 0
        for head in dataset.head_tail_rel_t["train"]:
            for tail in dataset.head_tail_rel_t["train"][head]:
                # Update progress bar
                counter += 1
                if counter % increment == 0:
                    remaining -= increment
                    pbar.update(increment)
                if remaining < increment:
                    pbar.update(remaining)
                for relb in dataset.head_tail_rel_t["train"][head][tail]:
                    ts_b = dataset.head_tail_rel_t["train"][head][tail][relb]
                    for relh in dataset.head_rel_ts["train"][head]:
                        for t in dataset.head_rel_ts["train"][head][relh]:
                            if window_size > 0 and not(start_beginning_flag) and t < window_size:
                                continue
                            if not (relb, relh) in examples_ff: examples_ff[(relb, relh)] = {}
                            correct = dataset.is_true(head, relh, tail, t, "train")
                            distances = get_distances_in_window(t, ts_b, window_size)
                            dd = tuple(distances)
                            if len(distances) > 0:
                                blocked = False
                                if not relh == relb:
                                    # blocked by recurrency expects time point not distances, that is why we have t-distances[-1]
                                    blocked = dataset.blocked_by_recurrency_train(head, relh, tail, t-distances[-1], t, RR_OFFSET)
                                # compute the forget weight, the more recent t is, the higher
                                if not blocked:
                                    if not dd in examples_ff[(relb, relh)]: examples_ff[(relb, relh)][dd] = [0,0]
                                    if correct: examples_ff[(relb, relh)][dd][0] += 1 
                                    examples_ff[(relb, relh)][dd][1] += 1
    # clean up a bit by removing all rules that have no positive examples (makes return value significantly smaller)
    print(">>> filtering xy-rules in learn-input by removing rules with not positive examples")
    examples_filtered = {}
    for rule in examples_ff:
        examples = examples_ff[rule]
        examples_fixed = {}
        preds, cpreds = 0, 0
        for ds in examples:
            pp = tuple(examples[ds])
            examples_fixed[ds] = pp
            cpreds += pp[0]
            preds += pp[1]
        if cpreds > 0:
            examples_filtered[rule] = examples_fixed
    print('>>> done!')
    return examples_filtered   


def make_learn_input(dataset, window_size,  progressbar_percentage, RR_OFFSET):
    """
    This function generates the learn input for the model.
     the beta_hat scores are accumulated which saves place
    :param dataset: the needed dataset object for data preprocessing
    :param window_size: specifies, how far we should look back during learning
    :param progressbar_percentage: for the increment of the bar progress, set 0 if you want to increase the progress by each triple
    :return examples: a dictionary with the following structure:
    {(body_relation, head_relation) : { delta_t1 : (true predictions, all predictions),  delta_t2 : (true predictions, all predictions), ... }}
    """
    # TODO we need to be able to specify what kinds of rules we want here?
    print(">>> starting to collect the examples required to learn confidence functions for xy-rules (single setting)")
    examples = {} # return values 
    start_beginning_flag = True # this flag, if set on True we should start learning from the beginning, otherwise if the query timestamp >= window_size
    # count different triples which is required for enstimating required / remaining time
    num_of_triples = 0
    for head in dataset.head_tail_rel_t["train"]: num_of_triples += len(dataset.head_tail_rel_t["train"][head])

    increment = int(num_of_triples*progressbar_percentage) if int(num_of_triples*progressbar_percentage) >=1 else 1
    remaining = num_of_triples
    # collect the examples
    # initialize tqdm progress bar with the total number of triples
    with tqdm(total=num_of_triples, desc=">>> collecting examples for xy-rules", unit="triple") as pbar:
        counter = 0
        for head in dataset.head_tail_rel_t["train"]:
            for tail in dataset.head_tail_rel_t["train"][head]:
                counter += 1
                if counter % increment == 0:
                    remaining -= increment
                    pbar.update(increment)
                if remaining < increment: pbar.update(remaining)
                for relb in dataset.head_tail_rel_t["train"][head][tail]:
                    ts = dataset.head_tail_rel_t["train"][head][tail][relb]
                    pointer = 0
                    j = ts[pointer] # the latest observation in the past
                    k = -1 # special index, informs about the closest body groduning of the trivial rec. rule whoch would also entail the prediction
                    i = ts[pointer] + 1 # the next time step
                    pointer += 1
                    while i <= dataset.latest_timestep:
                        # do something with i, make predictions for i if possible
                        d = i - j # d is the temporal distance between the observation (body grounding) and the prediction (rule head grounding)
                        if (d < window_size or window_size <= 0) and (start_beginning_flag or i >= window_size or window_size <= 0):
                            for relh in dataset.head_rel_ts["train"][head]:
                                if i in dataset.head_rel_ts["train"][head][relh]:
                                    # check if the prediction is true
                                    correct = dataset.is_true(head, relh, tail, i, "train")
                                    blocked = False
                                    if not relh == relb:
                                        blocked = dataset.blocked_by_recurrency_train(head, relh, tail, j, i, RR_OFFSET)
                                    if not blocked:
                                        if not (relb, relh) in examples: examples[(relb, relh)] = {}
                                        if not d in examples[(relb, relh)]: examples[(relb, relh)][d] = (0,0)
                                        examples[(relb, relh)][d] = (examples[(relb, relh)][d][0] + (1 if correct else 0), examples[(relb, relh)][d][1] + 1)
                        # increase i and check if there is another latest observation
                        i += 1
                        if pointer < len(ts) and i > ts[pointer]:
                            j = ts[pointer]
                            pointer += 1
    print('>>> done!')
    return examples


def extend_learn_input_constants_multi(dataset, previous_examples, options, progressbar_perc):
    # for the windows: do we start at the beginngin, even if smaller than window size? if True: yes, if False: no
    startlearntime = timeit.default_timer()
    
    WINDOW_SIZE = options['LEARN_WINDOW_SIZE']
    
    threshold_x_count = options["C_X_COUNT"]
    c_rule_recurrency_active = options["C_RULE_RECURRENCY_ACTIVE"]
    threshold_conf = options["C_THRESHOLD_CONFIDENCE"]
    
    RR_OFFSET = options["RR_OFFSET"]
    NEG_MINUS1 = RR_OFFSET >= -1    
    
    print(">> collecting the examples required as input to learn confidence functions for c-rules (multiple body groundings with time window)")

    # this is used for debugging print outs only
    # node_mapping, rel_mapping = read_id_to_string_mappings("E:/code/p-workspace/GraphTRuCoLa/tgb/datasets/tkgl_yago")
    
    # shot cuts for the datastraucres used within the following code
    head_tail_rel_t = dataset.head_tail_rel_t["train"]
    head_tail_rel_ts = dataset.head_tail_rel_ts["train"] # ts means time step set (its a set)
    head_rel_ts = dataset.head_rel_ts["train"]
    head_rel_t = dataset.head_rel_t["train"]
    

    c_rules = {}
    rbcb2rhch = precheck_sampling(c_rules, dataset, progressbar_perc, WINDOW_SIZE, threshold_x_count, c_rule_recurrency_active)

    num_of_ents = len(head_tail_rel_t)
    remaining = num_of_ents
    increment = int(num_of_ents*progressbar_perc) if num_of_ents*progressbar_perc >=1 else 1
    counter = 0


    # now do the real work, however, look only at those rules in detail that have more than threshold_x_count different x values as groundings
    example_count = {}
    with tqdm(total=num_of_ents, desc=">>> collecting examples for c-rules", unit="x-entities") as pbar:
        for x in head_tail_rel_t:
            counter += 1
            if counter % increment == 0:
                remaining -= increment
                pbar.update(increment)

            for cb in head_tail_rel_t[x]:    
                for rb in head_tail_rel_t[x][cb]:
                    if (rb,cb) in rbcb2rhch:              
                        for (rh,ch) in rbcb2rhch[(rb,cb)]:
                            backward_rule_key = (rh, ch, rb, cb, 'b') 
                            rule_key = (rh, ch, rb, cb) 
                            if not rule_key in example_count: example_count[rule_key] = 0
                            if not backward_rule_key in example_count: example_count[backward_rule_key] = 0
                            
                            ts_head_true = set()
                            if ch in head_tail_rel_ts[x] and rh in head_tail_rel_ts[x][ch]: ts_head_true = head_tail_rel_ts[x][ch][rh]
                            t_head_all = []
                            if rh in head_rel_t[x]: t_head_all = head_rel_t[x][rh] # atom exists Z with climbs(X,Z,T); e.g.: the timesteps where monkey1 (x) climbed somewhere
                            
                            # backward c-rules:
                            inv_rh = dataset.inverse_rel_dict[rh]

                            t_head_backward_all = []
                            if inv_rh in head_rel_t[ch]: t_head_backward_all = head_rel_t[ch][inv_rh] # atom exists Z with climbs(Z,CH,T) bec climbs(Z,CH,T) = invclimbs(CH,Z,T) e.g.: the timesteps where something climbed to level2 (ch)
                            if example_count[rule_key] < 10000:
                                (predictions, count) = get_c_rule_scores_x_multi(dataset, head_tail_rel_t[x][cb][rb], ts_head_true, t_head_all, False, NEG_MINUS1, x, WINDOW_SIZE)
                                update_c_rules_multi(c_rules, rule_key, predictions)
                                example_count[rule_key] += count
                            if example_count[backward_rule_key] < 10000:
                                (backward_predictions, count) = get_c_rule_scores_x_multi(dataset, head_tail_rel_t[x][cb][rb],ts_head_true, t_head_backward_all, False, NEG_MINUS1, x, WINDOW_SIZE)
                                update_c_rules_multi(c_rules, backward_rule_key, backward_predictions)
                                example_count[backward_rule_key] += count
    print(">>> created all c-rule learn data. now filter by confdence threshold")

    c_rules_filtered = {}
    forward_counter = 0
    backward_counter = 0
    rulekeys = list(c_rules.keys())
    for rule in rulekeys:
        examples = c_rules[rule]
        examples_fixed = {}
        preds, cpreds = 0, 0
        for ds in examples:
            pp = tuple(examples[ds])
            examples_fixed[ds] = pp
            cpreds += pp[0]
            preds += pp[1]
        if cpreds > 0 and cpreds / preds > threshold_conf:
            c_rules_filtered[rule] = examples_fixed
            if len(rule) == 4: forward_counter += 1
            if len(rule) == 5: backward_counter += 1
        del c_rules[rule]

    
    
    print(">>> collected examples (= learn input) for " + str(len(c_rules_filtered)) + " c-rules (forward=" + str(forward_counter) + ", backward=" + str(backward_counter) + "), done")
    
    previous_examples.update(c_rules_filtered)
    learntime = timeit.default_timer() -startlearntime
    print(">>> time for collecting examples for c-rules " + str(learntime) + "s")



def extend_learn_input_constants_single(dataset, previous_examples, options, progressbar_perc):
    """
    Extends the learn input, to contain also learn input (= examples) for rules with constants.
    These rules look like this: rel_h(X, constant1, T) <= rel_b(X, constant2, U)
    The key used for this extension is (rh, ch, rb, cb), the examples are stored in the usual SINGLE format.
    """
    # for the windows: do we start at the beginngin, even if smaller than window size? if True: yes, if False: no
    startlearntime = timeit.default_timer()
    print(">>> collecting the examples required as input to learn confidence functions for rules with constants, collecting rule candidates ...")
    
    WINDOW_SIZE = options['LEARN_WINDOW_SIZE']
    threshold_conf = options["C_THRESHOLD_CONFIDENCE"]
    threshold_x_count = options["C_X_COUNT"]
    c_rule_recurrency_active = options["C_RULE_RECURRENCY_ACTIVE"] #True
    RR_OFFSET = options["RR_OFFSET"]
    NEG_MINUS1 = RR_OFFSET >= -1
    
    # this is used for debugging print outs only
    # node_mapping, rel_mapping = read_id_to_string_mappings("E:/code/p-workspace/GraphTRuCoLa/tgb/datasets/tkgl_yago")
    
    # shot cuts for the datastraucres used within the following code
    head_tail_rel_t = dataset.head_tail_rel_t["train"]
    head_tail_rel_ts = dataset.head_tail_rel_ts["train"] # ts means time step set (its a set)
    head_rel_ts = dataset.head_rel_ts["train"]
    head_rel_t = dataset.head_rel_t["train"]
    
    c_rules = {}
    rbcb2rhch = precheck_sampling(c_rules, dataset, progressbar_perc, WINDOW_SIZE, threshold_x_count, c_rule_recurrency_active)

    num_of_ents = len(head_tail_rel_t)
    remaining = num_of_ents
    increment = int(num_of_ents*progressbar_perc) if num_of_ents*progressbar_perc >=1 else 1
    counter = 0

    # now do the real work, however, look only at those rules in detail that have more than threshold_x_count different x values as groundings
    example_count = {}
    with tqdm(total=num_of_ents, desc=">>> collecting examples (learn input) for c-rules", unit="entities as groundings of x") as pbar:
        for x in head_tail_rel_t:
            counter += 1
            if counter % increment == 0:
                remaining -= increment
                pbar.update(increment)
            for cb in head_tail_rel_t[x]:    
                for rb in head_tail_rel_t[x][cb]:
                    if (rb,cb) in rbcb2rhch:              
                        for (rh,ch) in rbcb2rhch[(rb,cb)]:
                            
                            # FORWARD:  rh(x,ch) <= rb(x,cb) & Ez rh(x,z)
                            # BACKWARD: rh(x,ch) <= rb(x,cb) & Ez rh(z,c) (last atom is equivalent to inv_rh(c,z) for which we have an index) 
                            inv_rh = dataset.inverse_rel_dict[rh]
                            rule_key = (rh, ch, rb, cb)
                            rule_key_backward = (rh, ch, rb, cb,'b')
                            if not rule_key in example_count: example_count[rule_key] = 0
                            if not rule_key_backward in example_count: example_count[rule_key_backward] = 0
                            ts_head_true = set()
                            t_head_all = []
                            if ch in head_tail_rel_ts[x] and rh in head_tail_rel_ts[x][ch]: ts_head_true = head_tail_rel_ts[x][ch][rh]
                            if example_count[rule_key] < 10000:
                                t_head_all = []
                                if rh in head_rel_t[x]: t_head_all = head_rel_t[x][rh] # collect timesteps where forward atom is true
                                (predictions, count) = get_c_rule_scores_x(dataset, head_tail_rel_t[x][cb][rb], ts_head_true, t_head_all, False, NEG_MINUS1, x, WINDOW_SIZE, RR_OFFSET)
                                example_count[rule_key] += count
                                update_c_rules(c_rules, rule_key, predictions)
                            if example_count[rule_key_backward] < 10000:
                                t_head_all_backward = []
                                if inv_rh in head_rel_t[ch]: t_head_all_backward = head_rel_t[ch][inv_rh] # collect timesteps where backward atom is true
                                (predictions_backward, count_backward) = get_c_rule_scores_x(dataset, head_tail_rel_t[x][cb][rb], ts_head_true, t_head_all_backward, False, NEG_MINUS1, x, WINDOW_SIZE, RR_OFFSET)
                                example_count[rule_key_backward] += count_backward
                                update_c_rules(c_rules, rule_key_backward, predictions_backward)

    c_rules_filtered = {}
    forward_counter = 0
    backward_counter = 0
    for rule in c_rules:
        examples = c_rules[rule]
        cpreds, preds = 0,0
        for d in examples:
            cpreds += examples[d][0]
            preds += examples[d][1]
        if preds > 0 and cpreds / preds > threshold_conf:
        # if preds > 0 and cpreds / preds > threshold_conf and cpreds > threshold_support:
            if len(rule) == 4: forward_counter += 1
            if len(rule) == 5: backward_counter += 1
            c_rules_filtered[rule] = examples
    
    print(">>> collected examples (= learn input) for " + str(len(c_rules_filtered)) + " c-rules (forward=" + str(forward_counter) + ", backward=" + str(backward_counter) + "), done")
    
    previous_examples.update(c_rules_filtered)
    learntime = timeit.default_timer() -startlearntime
    print(">>> time for collecting examples for c-rules " + str(learntime) + "s")



def precheck_sampling(c_rules, dataset, progressbar_perc, WINDOW_SIZE, threshold_x_count, c_rule_recurrency_active):

    c_rules_x_count = {}
    
    num_of_ts = len(dataset.time_head_rel_tails)
    remaining = num_of_ts
    increment = int(num_of_ts*progressbar_perc) if num_of_ts*progressbar_perc >=1 else 1
    counter = 0
    # print("MEM before: " +  str(process.memory_info().rss//1000000))
    # print("threshold_x_count: " +  str(threshold_x_count))
    # collect by sampling positive examples, due to the fact that we collect poistive examples we do not have to
    # distinguish between forward and backward direction odf the rule, the additional forward/backward atom is always true for a grounding that is a positive example 
    with tqdm(total=num_of_ts, desc=">>> sampling positive examples for c-rules", unit="timesteps") as pbar:
        for t in dataset.time_head_rel_tails:
            # print("MEM at " + str(t) + " " +  str(process.memory_info().rss//1000000), flush=True)
            counter += 1
            if counter % increment == 0:
                remaining -= increment
                pbar.update(increment)
            for head in dataset.time_head_rel_tails[t]:
                for relh in dataset.time_head_rel_tails[t][head]:
                    for tailh in dataset.time_head_rel_tails[t][head][relh]:
                        if not is_frequent_object_for_relation(dataset, relh, tailh, threshold_x_count): continue
                        tp_rand1 = t - random.randint(1,WINDOW_SIZE) # look back to a random time step within the window size
                        if WINDOW_SIZE > 1:
                            tp_rand2 = t - random.randint(1,WINDOW_SIZE // 2) 
                        else:
                            tp_rand2 = t - random.randint(1,WINDOW_SIZE // 2)
                        if WINDOW_SIZE > 5:
                            tp_rand5 = t - random.randint(1,WINDOW_SIZE // 5)
                        else: 
                            tp_rand5 = t - random.randint(1,WINDOW_SIZE // 2)
                        if WINDOW_SIZE > 10:
                            tp_rand10 = t - random.randint(1,WINDOW_SIZE // 10)
                        else:
                            tp_rand10 = t - random.randint(1,WINDOW_SIZE // 2)
                        tp_previous = t - 1  # look back to one step only
                        for tp_rand in (tp_rand1,tp_rand2,tp_rand5,tp_rand10,tp_previous):
                            if tp_rand > 0 and tp_rand in dataset.time_head_rel_tails:
                                if head in dataset.time_head_rel_tails[tp_rand]:
                                    for relb in dataset.time_head_rel_tails[tp_rand][head]:
                                        for tailb in dataset.time_head_rel_tails[tp_rand][head][relb]:
                                            rule_key = (relh, tailh, relb, tailb)
                                            if is_frequent_object_for_relation(dataset, relb, tailb, threshold_x_count):
                                                if not rule_key in c_rules_x_count: c_rules_x_count[rule_key] = set()
                                                if len(c_rules_x_count[rule_key]) <= threshold_x_count:
                                                    c_rules_x_count[rule_key].add(head)
    # hash the meaningfull head constants and head relations per rule body components
    rbcb2rhch = {}
    c_rule_cand_count = 0
    for k in c_rules_x_count:
        (rh, ch, rb, cb) = k
        if not (rb, cb) in rbcb2rhch: rbcb2rhch[(rb, cb)] = []
        if len(c_rules_x_count[k]) >= threshold_x_count:
            if not c_rule_recurrency_active and rh == rb and ch == cb: continue
            c_rule_cand_count += 1
            rbcb2rhch[(rb, cb)].append((rh,ch))
    print(">>>  after precheck 2 x " + str(c_rule_cand_count) + " = " + str(2 * c_rule_cand_count) + " c-rules candidates have been selected")
    return rbcb2rhch


def is_frequent_object_for_relation(dataset, rel, obj, threshold):
    rhtt = dataset.rel_head_tail_t["train"]
    rel_inv = dataset.get_inv_rel_id(rel)
    if rel_inv in rhtt:
        if obj in rhtt[rel_inv]:
            if len(rhtt[rel_inv][obj]) >= threshold // 5:
                return True
    return False

def update_c_rules(c_rules, rule_key, predictions):
    """ helper function for extend_learn_input_constants_single and extend_learn_input_constants_multi
    in the multi case delta does not refer to a single value but to a tuple as (2,5,19)
    """
    if not rule_key in c_rules: c_rules[rule_key] = {}
    for delta in predictions:
        if not delta in c_rules[rule_key]: c_rules[rule_key][delta] = [0,0]
        c_rules[rule_key][delta][0] += predictions[delta][0]
        c_rules[rule_key][delta][1] += predictions[delta][1]


def update_c_rules_multi(c_rules, rule_key, predictions):
    """ helper function for extend_learn_input_constants_single and extend_learn_input_constants_multi
    in the multi case delta does not refer to a single value but to a tuple as (2,5,19)
    """
    if not rule_key in c_rules: c_rules[rule_key] = {}
    for delta in predictions:
        if not delta in c_rules[rule_key]: c_rules[rule_key][delta] = [0,0]
        c_rules[rule_key][delta][0] += predictions[delta][0]
        c_rules[rule_key][delta][1] += predictions[delta][1]


def get_c_rule_scores_x(dataset, body_t, head_ts, head_exists_t, show, neg_minus1, x, WINDOW_SIZE, RR_OFFSET):
    """
    helper function for extend_learn_input_constants_single
    This function collects the examples for a single entity that has been used as
    x substitution in the current rule. Some of the parameters are lists some are sets.
    Its very important that it is like this! Lists are always ordered.
    
    :param body_t: A list of time steps for which the body of the rule is true for that specific x. 
    :param head_ts: A set of time steps for which the head of the rule is true for that specific x.  
    :param head_exists_t: A list of time steps for which something is stated w.r.t to x and the head relation of the rule.  
    :param show: If set to True some debugging prints are created. Not used in normal mode.
    :param neg_minus1: If this is true a negated atom is added to the body, if h(X,c, t) is the head, then !h(X,c, t-1) is the additional body atom.
    :param x: The x entity for which data is collected (used for debugging prupose only).
    :param WINDOW_SIZE: See options parameter named LEARN_WINDOW_SIZE
    :return predictions: delta_t1 : (true predictions, all predictions),  delta_t2 : (true predictions, all predictions), ... }
    """
    # some lines are left for debugging purpose which create prints when show is set to True
    index_b = 0 
    index_h = 0 
    predictions = {}
    example_counter = 0
    while index_h < len(head_exists_t) and head_exists_t[index_h] <= body_t[index_b]:
        index_h += 1
    while index_b < len(body_t):
        bt_current = body_t[index_b]
        bt_next =  body_t[index_b+1] if index_b+1 < len(body_t) else 10000000 # hope thats high enough, horrible coding ...
        while index_h < len(head_exists_t) and head_exists_t[index_h] <= bt_next:
            ht = head_exists_t[index_h]
            # everthing fine in bt_current and ht
            delta = ht - bt_current
            if delta < WINDOW_SIZE and (not(neg_minus1) or not(ht-1 in head_ts)):
                if not blocked_by_recurrency(ht, delta, head_ts, RR_OFFSET):
                    if not delta in predictions: predictions[delta] = [0,0]
                    if ht in head_ts: predictions[delta][0] += 1
                    predictions[delta][1] += 1
                    example_counter += 1
            index_h += 1
        if index_h == len(head_exists_t): break
        index_b += 1
    return (predictions, example_counter)


def blocked_by_recurrency(ht, delta, head_ts, RR_OFFSET):
    """
    Helper function which is hard to explain ...
    
    :param ht: Time step for which the body is true. Time step that is precicted
    :param delta: Time distance for which the body is true w.r.t ht. Distance to the time step from which is predicted.
    :param head_ts: A set of time steps for which the head of the rule is true. 
    :param RR_OFFSET: Some offset.
    :return True if the prediction is blocked, false otherwise 
    """
    if RR_OFFSET < 0: return False
    for d in range(1, delta + RR_OFFSET + 1):
        if ht - d in head_ts: return True
    return False


def get_c_rule_scores_x_multi(dataset, body_t, head_ts, head_exists_t, show, neg_minus1, x, WINDOW_SIZE):
    """
    helper function for extend_learn_input_constants_multi
    
    This function collects the examples for a single entity that has been used as x substitution in the current rule.
    Some of the parameters are lists some are sets.  Lists are always ordered.
    
    :param body_t: A list of time steps for which the body of the rule is true for that specific x. 
    :param head_ts: A set of time steps for which the head of the rule is true for that specific x.  
    :param head_exists_t: A list of time steps for which something is stated w.r.t to x and the head relation of the rule.  
    :param show: If set to True some debugging prints are created. Not used in normal mode.
    :param neg_minus1: If this is true a negated atom is added to the body, if h(X,c, t) is the head, then !h(X,c, t-1) is the additional body atom.
    :param x: The x entity for which data is collected (used for debugging prupose only).
    :param WINDOW_SIZE: See options parameter named LEARN_WINDOW_SIZE
    :return predictions: delta_t1 : (true predictions, all predictions),  delta_t2 : (true predictions, all predictions), ... }
    """
    # some lines are left for debugging purpose which create prints when show is set to True
    index_b = 0 
    index_h = 0 
    predictions = {} 
    example_counter = 0
    while index_h < len(head_exists_t) and head_exists_t[index_h] <= body_t[index_b]:
        index_h += 1
    while index_b < len(body_t):
        bt_current = body_t[index_b]
        bt_next =  body_t[index_b+1] if index_b+1 < len(body_t) else 10000000 # hope thats high enough, horrible coding ...
        while index_h < len(head_exists_t) and head_exists_t[index_h] <= bt_next:
            ht = head_exists_t[index_h]
            # everthing fine in bt_current and ht
            delta = ht - bt_current
            if delta < WINDOW_SIZE and (not(neg_minus1) or not(ht-1 in head_ts)):
                distances = get_distances_in_window(ht, body_t[:index_b+1], WINDOW_SIZE)
                tdistances = tuple(distances)
                if not tdistances in predictions:
                    predictions[tdistances] = [0,0]
                if ht in head_ts: predictions[tdistances][0] += 1
                predictions[tdistances][1] += 1
                example_counter += 1
            index_h += 1
        if index_h == len(head_exists_t): break
        index_b += 1
    return (predictions, example_counter)


def get_distances_in_window(t, ts_b, window_size):
    """
    t is the max timestep - the timesteps should be smaller than this one to be added
    ts_b is a list of timesteps. we check which of them is in window
    window_size is the size of the window
    """
    # TODO speed up this method by adding a binary search to find the index of the highes tb in ts_b which is smaller than t
    # go back from this index until you are out of the window_size
    distances = []
    for tb in ts_b:
        if t > tb:
            if t - tb < window_size or window_size <= 0:
                distances.append(t - tb)
    return distances