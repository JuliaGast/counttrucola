import csv
import os
import sys


dataset = 'icews14'
dataset_name = 'tkgl_'+dataset
dataset_name2 = 'tkgl-'+dataset

if 'gdelt' in dataset:
    ts_size = 15
elif 'icews14' in dataset or 'icews18' in dataset: 
    ts_size = 24
else:
    ts_size = 1

print("Current sys.path:", sys.path)

# File paths
edgelist_path = os.path.join(sys.path[0], dataset_name, dataset_name2+'_edgelist.csv')
rel_mapping_path = os.path.join(sys.path[0],dataset_name, 'rel_mapping.csv')
node_mapping_path = os.path.join(sys.path[0],dataset_name, 'node_mapping.csv')


type_of_output = 'tgbid'  # or 'string' Type of output to generate

include_inverse_flags = True  # Set to True if you want to include inverse relations


if type_of_output == 'tgbid':
    node_index = 1
    rel_index = 0
    output_string_name = 'tgbid_edgelist.txt'
elif type_of_output == 'string':
    node_index = 2
    rel_index = 1
    output_string_name = 'string_edgelist.txt'
if include_inverse_flags:
    # ruledataset = RuleDataset(name=dataset_name2, threshold=1, large_data_hardcode_flag=False)
    output_string_name = 'incl_inverse_' + output_string_name


output_path = os.path.join(sys.path[0],dataset_name, output_string_name)


sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from rule_based.rule_dataset import RuleDataset 



if include_inverse_flags:
    ruledataset = RuleDataset(name=dataset_name2, threshold=1, large_data_hardcode_flag=False)
    # output_string_name = 'incl_inverse_' + output_string_name



max_line = 90730 #2278405 # 610153
min_line = 0

# Load mappings
with open(rel_mapping_path, 'r', encoding='utf-8') as f:
    rel_mapping = {int(row[0]): row[rel_index] for i, row in enumerate(csv.reader(f, delimiter=';')) if i > 0}

with open(node_mapping_path, 'r', encoding='utf-8') as f:
    node_mapping = {int(row[0]): row[node_index]  for i, row in enumerate(csv.reader(f, delimiter=';')) if i > 0}

# Process edgelist and write output
with open(edgelist_path, 'r') as edgelist_file, open(output_path, 'w') as output_file:
    reader = csv.reader(edgelist_file)
    for i, row in enumerate(reader):
        if i >= max_line:
            break
        if i > min_line:

            timestep, head, tail, rel = map(int, row) # ts,head,tail,relation_type
            head_str = node_mapping.get(head, f'unknown_{head}')
            rel_str = rel_mapping.get(rel, f'unknown_{rel}')
            tail_str = node_mapping.get(tail, f'unknown_{tail}')
            timestep = timestep // ts_size  # Adjust timestep based on ts_size
            output_file.write(f"{head_str}\t{rel_str}\t{tail_str}\t{timestep}\n")

            if include_inverse_flags:
                if type_of_output == 'tgbid':
                    inverse_rel = ruledataset.get_inv_rel_id(rel)
                    
                elif type_of_output == 'string':
                    inverse_rel = 'inv_' + rel_str
                output_file.write(f"{tail_str}\t{inverse_rel}\t{head_str}\t{timestep}\n")

print(f"Processed {i - min_line} lines from the edgelist and saved to {output_path}")