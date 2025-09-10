import math
from learner import ParamLearner
from rules import Rule1, Rule2, RuleC, RuleCBackward

def filter_and_learn_params(learninput, ruleset_noparams, stats_file_path, dataset_name, rel_mapping, node_mapping, options, figure_path, progressbar_percentage=0.1):
    """set the parameters for the confidence functions - select what to do based on the option specified in learn_option
    :return: the parameters for the confidence functions with keys (rulekey) and values [lmbda, alpha, phi, rho, kappa, gamma, function] 
    :return: mse_curvefit [float] the mean squared error of the curve fit, 0 if no curve_fit was done
    """
    mse_curvefit = 0
    learn_option = options["LEARN_PARAMS_OPTION"]
    static_flag= False

    if learn_option == 'learn':
        learn_flag = True
    elif learn_option =='compute':
        learn_flag = False
    elif learn_option == 'static':
        learn_flag = False
        static_flag = True
    else:
        print(f'option {learn_option} not implemented yet. using default parameters ')
        params = default_parameters(ruleset_noparams, options=options)
        learn_option = 'default' 
        return params, learn_option, mse_curvefit


    datapoint_threshold_multi = options['DATAPOINT_THRESHOLD_MULTI'] 

    learner = ParamLearner(dataset_name, rel_mapping, node_mapping, options['PLOT_FLAG'], figure_path, multi_flag=options['MULTI_FLAG'], single_flag=options['SINGLE_FLAG'], rule_unseen_neg=options['RULE_UNSEEN_NEGATIVES'], 
                           learn_flag=learn_flag, lmbda_reg=options['LMBDA_REG'], datapoint_threshold_multi=datapoint_threshold_multi,
                           static_flag=static_flag, max_time_window=options['LEARN_WINDOW_SIZE'])
    
    learninput = filter_learninput(learninput, ruleset_noparams, options=options)
    learner.learn_data = learninput
    params, mse_curvefit = learner.learn_params(progressbar_percentage)

    return params,mse_curvefit


def filter_learninput(learninput, ruleset_noparams,  options):
    """ filter the learninput to only include the rules in ruleset_noparams
    also exclude rules with (static) confidence below threshold_conf and correct predictions below threshold_correct_preds
    :param learninput: [dict] the  learninput with keys (relb,relh)
    :param_ruleset_noparams: the ruleset that was created before based on the config (reccurency rules only? include cyclic rules?)
    :param options: [dict] the options from the config file
    :return: learninput_new [dict] the filtered learninput with keys (relb,relh), 

    """
    rule_list = []
    listrules = ruleset_noparams.rules
    for rule in listrules:
        rulekey = rule.get_rule_key()
        rule_list.append(rulekey)
    set_ruleset_noparams = set(rule_list)

    ruleset_noparams.rules = []

    threshold_correct_preds = options["THRESHOLD_CORRECT_PREDICTIONS"]
    threshold_conf = options["THRESHOLD_CONFIDENCE"]

    learninput_new = {}
    num_rules = len(set_ruleset_noparams)
    num_accepted_rules = 0


    for rule_key in learninput:
        if rule_key not in set_ruleset_noparams:
            continue
        preds = 0
        predsc = 0 
        for d in learninput[rule_key]:
            (pc,p) = learninput[rule_key][d]
            preds += p
            predsc += pc

        confidence = (predsc / preds)
       
                

        if predsc > threshold_correct_preds and (confidence > threshold_conf):
            num_accepted_rules += 1
            learninput_new[rule_key] = learninput[rule_key]
            if len(rule_key) == 2:
                (relb,relh) = rule_key
                if relb == relh:
                    ruleset_noparams.rules.append(Rule1(relb, [], ruleset_noparams.rel_id_to_string[relb]))
                else:
                    ruleset_noparams.rules.append(Rule2(relh, relb, [], ruleset_noparams.rel_id_to_string[relh], ruleset_noparams.rel_id_to_string[relb]))
            if len(rule_key) == 4:
                (rh, ch, rb, cb) = rule_key
                ruleset_noparams.rules.append(RuleC(rh, ch, rb, cb, [], ruleset_noparams.rel_id_to_string[rh], ruleset_noparams.node_id_to_string[ch][0], ruleset_noparams.rel_id_to_string[rb], ruleset_noparams.node_id_to_string[cb][0]))
            if len(rule_key) == 5:
                (rh, ch, rb, cb, b) = rule_key
                ruleset_noparams.rules.append(RuleCBackward(rh, ch, rb, cb, [], ruleset_noparams.rel_id_to_string[rh], ruleset_noparams.node_id_to_string[ch][0], ruleset_noparams.rel_id_to_string[rb], ruleset_noparams.node_id_to_string[cb][0]))

                
    num_removed_rules = num_rules - num_accepted_rules
    try:
        percentage = num_removed_rules / num_rules
    except:
        percentage = 0.0
    print(f">>> {percentage * 100:.2f}% of the learn input rules were removed due confidence restriction, therefore the number of the rules was reduced from {num_rules} to {num_accepted_rules}.")
            
    return learninput_new


def compute_static_parameters(learninput, ruleset_noparams):
    """ compute the parameters based on the static confidence
    :param learninput: [dict] the  learninput with keys (rulekey), and values dict with  time distances and values list of tuples (correct_preds, preds)
    :param ruleset_noparams: [Ruleset] the ruleset without parameters
    :returns: parameters [dict] the parameters for the confidence functions with keys (rulekey), and values list of floats [lmbda, alpha, phi, rho, kappa, gamma, function], 
    where only alpha gets a nonzero value  [0.0, confidence, 0.0, 0.0, 0.0, 0.0, "powtwo"]
    """
    # we only need to find params for the rules i ruleset_noparams - these are created based on flags specified in config file
    rule_list = []
    listrules = ruleset_noparams.rules
    for rule in listrules:
        rule_key = rule.get_rule_key()
        rule_list.append(rule_key)
    set_ruleset_noparams = set(rule_list)

    parameters = {}
    num_accepted_rules = 0

    for rule in learninput:
        if rule not in set_ruleset_noparams:
            continue # then we do not need any parameters for this rule
        preds = 0
        predsc = 0 
        for d in learninput[rule]:
            (pc,p) = learninput[rule][d]
            preds += p
            predsc += pc

        confidence = (predsc / preds)


       
        num_accepted_rules += 1

        parameters[rule] = [0.0, confidence, 0.0, 0.0, 0.0, 0.0, "powtwo"]

    print(f">>> static confidence parameters [lmbda=0, alpha=static_conf, phi=0, rho=0, kappa=0, gamma=0, function]  were assigned to all rules, in total", num_accepted_rules, "rules")
    return parameters

def default_parameters(ruleset_noparams, options):
    """
    :param options: [dict] the options from the config file
    """

    default_lmbda = 1
    default_alpha = 0.5
    default_phi = 0.1
    default_func = 'powtwo'
    default_kappa = 0
    default_rho = 0.01
    default_gamma = 0.01

    default_parameters = [default_lmbda, default_alpha, default_phi, default_rho, default_kappa, default_gamma, default_func]
    parameters = {}
    counter = 0
    for rule in ruleset_noparams.rules:
        rule_key = rule.get_rule_key()
        parameters[rule_key] = default_parameters
        counter += 1

    print(f">>> default parameters {default_parameters} [lmbda, alpha, phi, rho, kappa, gamma, function] were assigned to all rules, in total", counter, "rules")
   
    return parameters

