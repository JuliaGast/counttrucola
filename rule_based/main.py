import timeit 
import argparse

from options import Options
from eval import evaluate
from rule_dataset import RuleDataset
import confidence_params
import rule_utils
import predictor
import utils
import learn_input_creator



def main(options_call=None): 
    start_time = timeit.default_timer()
    print("starting at: ", start_time)
    parser = argparse.ArgumentParser()

    ## preparation
    # config stuff
    parser.add_argument("--config", type=str, default="config-default.yaml", help="Path to the configuration file")
    args, _ = parser.parse_known_args()    
    config_file_name = args.config    

    options_obj = Options(config_file_name=config_file_name, options_call=options_call)
    options = options_obj.options

    print("options: ", options)

    
    # dataset
    dataset_name = options["DATASET_NAME"]
    # create the rule dataset
    ruledataset = RuleDataset(name=dataset_name, large_data_hardcode_flag=options["LARGE_DATA_HARDCODE_FLAG"], very_large_data_hardcode_flag=options["VERY_LARGE_DATA_HARDCODE_FLAG"])

    # small and medium sized 
    if not(ruledataset.large_data_flag) and not(ruledataset.very_large_data_flag):
        if not "C_X_COUNT" in options: options["C_X_COUNT"] = 3
    # large 
    if ruledataset.large_data_flag and not(ruledataset.very_large_data_flag):
        if not "C_X_COUNT" in options: options["C_X_COUNT"] = 20
    if ruledataset.very_large_data_flag:
        # we will check this param when its about creating c-rules, c-rule will not be generated for very large datasets
        if not "C_X_COUNT" in options: options["RULE_TYPE_C"] = False
    
    
    # sanity check for window size.
    ruledataset.sanity_check_window_size(options['LEARN_WINDOW_SIZE'])

    # define and create all necessary directories
    learn_data_path, results_path, figure_path, stats_file_path, log_path, path_rankings_test, path_rankings_val, params_dir, results_dir, rules_dir, path_rules, all_rule_types_false = utils.sort_out_pathnames(options, dataset_name)

    # create learn data = collect examples; section 4.2 Collecting Examples 
    startlearndatatime= timeit.default_timer()  
    learn_input = learn_input_creator.load_or_make_learn_input(options, learn_data_path, ruledataset)
    learndatatime = timeit.default_timer() -startlearndatatime


    # create the ruleset for the rules of interest; rules are described in section 4.1 Rules
    ruleset_noparams = rule_utils.generate_ruleset(ruledataset, options)
    if options["RULE_TYPE_C"]: rule_utils.extend_ruleset_constants(ruleset_noparams, learn_input, ruledataset, options)

    ## rule learning: learn or load the params for confidence functions - depending on learn_option; section 4.2 Confidence Functions and  Learning the Confidence Functions
    print("-----------------fetch or learn params for config functions, option: ", options["LEARN_PARAMS_OPTION"])   
    if options["LOAD_PARAMS_FLAG"]:
        print('load the parameters for the confidence functions from file: ', path_rules.format("ids"))       
        learntime, mse_curvefit=0,0
    else:
        startlearntime= timeit.default_timer()  
        # learn the paramters for the confidence functions - before that, filter the rules acc. to thresholds   
        params, mse_curvefit = confidence_params.filter_and_learn_params(learn_input, ruleset_noparams, stats_file_path,  dataset_name, 
                                                                ruledataset.rels_id_to_string, ruledataset.nodes_id_to_string, options, figure_path, 
                                                                progressbar_percentage=0.01)
        # add the params to the so far empty ruleset
        ruleset_noparams.add_params(params) 
        learntime = timeit.default_timer() -startlearntime
        print('time to learn or set parameters: ', learntime)
        rule_utils.write_rule_files(path_rules, ruleset_noparams)

    ## apply: read rules and apply them  to the given dataset - aggregation: section 4.3 
    predictor.setOptions(options)
    startapplytime = timeit.default_timer()
    if options["APPLY_RULES_FLAG"]:
        print('-----------------apply the rules to the dataset')
        if options['EVAL_TESTSET_FLAG']:
            number_of_rules = predictor.apply(ruledataset, path_rules.format("ids"), path_rankings_test, progressbar_percentage=0.001, evaluation_mode='test', all_rule_types_false=all_rule_types_false)
        if options["EVAL_VALSET_FLAG"]:
            number_of_rules = predictor.apply(ruledataset, path_rules.format("ids"), path_rankings_val, progressbar_percentage=0.001, evaluation_mode='val', all_rule_types_false=all_rule_types_false)
    else:
        try:
            number_of_rules = predictor.read_number_of_rules(ruledataset, path_rules.format("ids"))
        except Exception as e:
            print("Error reading number of rules:", e)
            number_of_rules = 0
    applytime = timeit.default_timer() - startapplytime
    print('time to apply the rules: ', applytime)

    ## evaluate: compute the MRR and hits scores based on the computed rankings; experimental setup: section 5.1
    testmrr, testhits10, testhits1,testhits100, valmrr, valhits10, valhits1, valhits100 = -1,-1,-1,-1,-1,-1, -1,-1
    valmrrperrel,valhits1perrel,  testmrrperrel, testhits1perrel = {}, {}, {}, {}
    startevaltime = timeit.default_timer()
    if options['EVAL_TESTSET_FLAG']:
        print('-----------------compute the test MRR, using path rankings from ', path_rankings_test)  
        testmrr, testhits10, testhits1, testhits100, testmrrperrel, testhits1perrel, testmrrperts, testhits1perts =evaluate(ruledataset, path_rankings_test, progressbar_percentage=0.01, evaluation_mode='test', eval_type=options['EVAL_TYPE'])
    if options["EVAL_VALSET_FLAG"]:
        print('-----------------compute the val MRR, using path rankings from ', path_rankings_val)  
        valmrr, valhits10, valhits1, valhits100, valmrrperrel,valhits1perrel, valmrrperts, valhits1perts =evaluate(ruledataset, path_rankings_val, progressbar_percentage=0.01, evaluation_mode='val', eval_type=options['EVAL_TYPE'])
    evaltime =  timeit.default_timer()- startevaltime
    print('eval time: ', evaltime )
    endtime = timeit.default_timer()
    totaltime = endtime - start_time
    print('-----------------done! It took ', totaltime, ' seconds')  


    ## logging: write the config and results to file
    utils.write_config_and_results(results_path, options, dataset_name, path_rankings_test, testmrr, testhits100, testhits10, testhits1, 
                             valmrr, valhits100, valhits10, valhits1, number_of_rules, mse_curvefit, ruledataset.large_data_flag, ruledataset.very_large_data_flag,
                             totaltime, evaltime, applytime, learntime, learndatatime)
    if options["EVAL_TESTSET_FLAG"]:
        utils.write_ranksperrel(testmrrperrel, testhits1perrel, results_dir, dataset_name, 'test', options['Z_RULES_FACTOR'])
    if options["EVAL_VALSET_FLAG"]:
        utils.write_ranksperrel(valmrrperrel, valhits1perrel, results_dir, dataset_name, 'val', options['Z_RULES_FACTOR'])


    if options["DELETE_RANKINGS_FLAG"]:
        print('delete the rankings files')
        utils.delete_rankings(path_rankings_test)
        utils.delete_rankings(path_rankings_val)

    return valmrr


if __name__ == "__main__":
    val_mrr = main()  # you can set the learn window size here, if you want to test it