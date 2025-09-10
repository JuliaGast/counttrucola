import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__)))) 
print(sys.path)
from utils.utils import fetch_wikidata_property_name
import argparse
from tqdm import tqdm

args = argparse.ArgumentParser() 
args.add_argument("--dataset_name", "-d", default="tkgl-smallpedia", help="Name of the dataset") 
args = args.parse_args()

dataset_name = args.dataset_name

dataset_name2 = dataset_name.replace('-', '_')

string_dict ={}
root = osp.dirname(osp.abspath(__file__))

input_dir = osp.join(root, dataset_name2, dataset_name+'_edgelist.csv')

rel_2_string = {}
entity_2_string = {}

# open csv file and read it line by line
lines =[]
with open(input_dir, 'r', encoding='utf-8') as file:
    # with tqdm iterates over the file and shows a progress bar
    for line_r in tqdm(file):
    # for line_r in file:
        parts = line_r.strip().split(',')
        if len(parts) < 3:
            continue
        if 'smallpedia' in dataset_name or 'wikidata' in dataset_name:
            head, rel, tail = parts[1], parts[3], parts[2]
        else:
            head, rel, tail = parts[0], parts[1], parts[2]
        if head == 'head':
            continue

        if rel not in rel_2_string:
            rel_2_string[rel] = fetch_wikidata_property_name(rel)

        if head not in entity_2_string:
            entity_2_string[head] = fetch_wikidata_property_name(head)

        if tail not in entity_2_string:
            entity_2_string[tail] = fetch_wikidata_property_name(tail)

# write entity2id.txt and relation2id.txt file with key \t 1 \t value from the dictionary
with open(osp.join(root, dataset_name2, 'entity2string.txt'), 'w', encoding='utf-8') as file:
    for key, value in entity_2_string.items():
        file.write(f"{key}\t{value}\n")

with open(osp.join(root, dataset_name2, 'relation2string.txt'), 'w', encoding='utf-8') as file:
    for key, value in rel_2_string.items():
        file.write(f"{key}\t{value}\n")


print('done')