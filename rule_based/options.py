import yaml
import flatdict
import copy
import os
import argparse


class Options():
    def __init__(self, path=None, config_file_name="config-default.yaml", options_call=None):
        self.options = {}

        default_path = os.path.join("rule_based",config_file_name)
        with open(default_path, 'r') as file:
            raw_config  = yaml.safe_load(file)

        # 1. DEFAULT: Start with whole top-level keys EXCEPT DATASET_OVERRIDES
        base_options = {}
        for k, v in raw_config.items():
            if k not in ("DATASET_OVERRIDES"):
                base_options[k] = v

        # 2. Parse the options from -- params
        parsed_options_dict = self.parse_options() # here i only need the param "DATASET_NAME" for now. all other params will be set later
        print("parsed_options_dict: ", parsed_options_dict)

        if options_call is not None and "DATASET_NAME" in options_call:
            dataset_name = options_call["DATASET_NAME"]
            print(f"Using dataset name from options_call: {dataset_name}")
        else:
            dataset_name = base_options.get("DATASET_NAME") # otherwise use the dataset_name that was speficied in main (options_call)
            if "DATASET_NAME" in parsed_options_dict: # if this has been set by command line
                dataset_name = parsed_options_dict["DATASET_NAME"]
            print(f"Using dataset name from base options: {dataset_name}")

        # 3. OVERWRITE: Merge DATASET_OVERRIDES from config file, if present and matching
        overrides = raw_config.get("DATASET_OVERRIDES", {})
        print('dataset_name: ', dataset_name)
        print("DATASET_OVERRIDES: ", overrides)
        if dataset_name and dataset_name in overrides:
            print(f"Found dataset overrides for {dataset_name}")
            for k, v in overrides[dataset_name].items():
                print(f"Overwriting {k} with {v} from DATASET_OVERRIDES for dataset {dataset_name}")
                base_options[k] = v

        for key, value in base_options.items():
            if not isinstance(value, dict):
                self.options[key] = value

        # 4. OVERWRITE: Extra override file (your existing logic)
        # if path:
        #     options_extra = {}
        #     with open(path, 'r') as file:
        #         print(">>> using options in " + path + " to overwrite some default options")
        #         options_extra  = yaml.safe_load(file)

        #     options_extra_flat = dict(flatdict.FlatDict(options_extra, delimiter='.'))
        #     for param in options_extra_flat:
        #         self.set(param, options_extra_flat[param])

        # 5. OVERWRITE: overwrite from parsed command line options
        print('overwriting options from config file and command line call with parsed options')
        for key, value in parsed_options_dict.items():
            print(f"Overwriting {key} with {value} from command line")
            self.set(key, value)

        # 6. OVERWRITE: If options_call is provided, overwrite the options with it
        if options_call is not None:
            print('overwriting options from config file and command line call with options_call that were passed to main()')
            for key, value in options_call.items():
                print(f"Overwriting {key} with {value}")
                self.set(key, value)
                overrides = raw_config.get("DATASET_OVERRIDES", {})
                if dataset_name and dataset_name in overrides:
                    for k, v in overrides[dataset_name].items():
                        self.set(k, v)


    def set(self, parameter, value):
        '''
        To set a new value for a specified parameter
        :param parameter: the parameter that has to be changed
        :param value: the valued that needs to be set
        '''
        param_bad = False
        if "." in parameter:
            ktoken = parameter.split(".")
            o = self.options
            for t in ktoken[:-1]:
                if t in o:
                    if type(o[t]) == dict: o = o[t]
                    else:
                        param_bad = True
                        break
                else:
                    param_bad = True
                    break
            if not param_bad:
                if ktoken[-1] in o:
                    o[ktoken[-1]] = value
                else:
                    param_bad = True
        else:
            if parameter in self.options:
                self.options[parameter] = value
            else:
                param_bad = True
        if param_bad:
            raise Exception(f"Trying to set key parameter {str(parameter)} in the options but the key does not exist.")



    def parse_options(self):
        parsed_options_dict ={}
        parser = argparse.ArgumentParser(description="You can parse the parameter that you want to change in the config file")
        parser.add_argument("--params", nargs='+', help="Key-value pairs in the format key=value")

        args, unknown_args = parser.parse_known_args()
        print(args)

        print('---------------------')
        # print(args.params)
        if type(args.params) == list:
            print("Parsing command line options:")
            for pair in args.params:
                key, value = pair.split("=")
                try:
                    # check if the value is integer and if so convert it
                    value = int(value)
                except ValueError:
                    try:
                        # check if the value is float and if so convert it
                        value = float(value)
                        
                    except ValueError:
                        # check if the value is boolean and if so convert it
                        if value == "True" or value == "False":
                            value = value == "True"
                        else:
                            pass
                print(f"Parsed option: {key} = {value}")
                # print(key, value)
                parsed_options_dict[key] = value
                # self.set(key, value)

        return parsed_options_dict

