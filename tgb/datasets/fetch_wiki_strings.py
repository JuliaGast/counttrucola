import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__)))) 
print(sys.path)
from utils.utils import fetch_wikidata_property_name
import argparse

args = argparse.ArgumentParser() 
args.add_argument("--dataset_name", "-d", default="tkgl-wikiold", help="Name of the dataset") 
args = args.parse_args()

dataset_name = args.dataset_name

dataset_name2 = dataset_name.replace('-', '_')

string_dict ={}
root = osp.dirname(osp.abspath(__file__))
input_dir = osp.join(root, dataset_name2, 'entity2id.txt')
input_dir_rels = osp.join(root, dataset_name2, 'relation2id.txt')
lines2 =[]
with open(input_dir, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        string, original_id = parts[0], int(parts[1])

        string2 = fetch_wikidata_property_name(string)
        string_dict[original_id] = string
        lines2.append([string, original_id, string2])

output_dir = osp.join(root, dataset_name2, 'new_entity2id.txt')

with open(output_dir, 'w', encoding='utf-8') as file:
    for line in lines2:
        string, original_id, string2 = line
        file.write(f"{string}\t{original_id}\t{string2}\n")

lines3 =[]
with open(input_dir_rels, 'r', encoding='utf-8') as file:
    for line_r in file:
        parts = line_r.strip().split('\t')
        if len(parts) < 2:
            continue
        string, original_id = parts[0], int(parts[1])

        string2 = fetch_wikidata_property_name(string)

        lines3.append([string, original_id, string2])

output_dir = osp.join(root, dataset_name2, 'new_rel2id.txt')

with open(output_dir, 'w', encoding='utf-8') as file:
    for line in lines3:
        string, original_id, string2 = line
        file.write(f"{string}\t{original_id}\t{string2}\n")


print('done')