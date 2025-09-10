import random
import os
import pickle
import sys
import argparse
import json
#import torch
from typing import Any
import numpy as np
#from torch_geometric.data import TemporalData
import pandas as pd
#import torch
import os.path as osp
import requests
import re


def add_inverse_quadruples(df: pd.DataFrame) -> pd.DataFrame:
    """
    adds the inverse relations required for the model to the dataframe
    """
    if ("edge_type" not in df):
        raise ValueError("edge_type is required to invert relation in TKG")
    
    sources = np.array(df["u"])
    destinations = np.array(df["i"])
    timestamps = np.array(df["ts"])
    edge_idxs = np.array(df["idx"])
    weights = np.array(df["w"])
    edge_type = np.array(df["edge_type"])

    num_rels = np.unique(edge_type).shape[0]
    inv_edge_type = edge_type + num_rels

    all_sources = np.concatenate([sources, destinations])
    all_destinations = np.concatenate([destinations, sources])
    all_timestamps = np.concatenate([timestamps, timestamps])
    all_edge_idxs = np.concatenate([edge_idxs, edge_idxs+edge_idxs.max()+1])
    all_weights = np.concatenate([weights, weights])
    all_edge_types = np.concatenate([edge_type, inv_edge_type])

    return pd.DataFrame(
            {
                "u": all_sources,
                "i": all_destinations,
                "ts": all_timestamps,
                "label": np.ones(all_timestamps.shape[0]),
                "idx": all_edge_idxs,
                "w": all_weights,
                "edge_type": all_edge_types,
            }
        )



def add_inverse_quadruples_np(quadruples: np.array, 
                              num_rels:int) -> np.array:
    """
    creates an inverse quadruple for each quadruple in quadruples. inverse quadruple swaps subject and objsect, and increases 
    relation id by num_rels
    :param quadruples: [np.array] dataset quadruples, [src, relation_id, dst, timestamp ]
    :param num_rels: [int] number of relations that we have originally
    returns all_quadruples: [np.array] quadruples including inverse quadruples
    """
    inverse_quadruples = quadruples[:, [2, 1, 0, 3]]
    inverse_quadruples[:, 1] = inverse_quadruples[:, 1] + num_rels  # we also need inverse quadruples
    all_quadruples = np.concatenate((quadruples[:,0:4], inverse_quadruples))
    return all_quadruples




# import torch
def save_pkl(obj: Any, fname: str) -> None:
    """
    save a python object as a pickle file
    """
    with open(fname, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(fname: str) -> Any:
    """
    load a python object from a pickle file
    """
    with open(fname, "rb") as handle:
        return pickle.load(handle)


# def set_random_seed(seed: int):
#     """
#     setting random seed for reproducibility
#     """
#     np.random.seed(seed)
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)

def set_random_seed(random_seed: int):
    """
    set random seed for reproducibility
    Args:
        random_seed (int): random seed
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except:
        pass
    print(f'INFO: fixed random seed: {random_seed}')



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_args():
    parser = argparse.ArgumentParser('*** TGB ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--num_epoch', type=int, help='Number of epochs', default=30)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_dim', type=int, help='Memory dimension', default=100)
    parser.add_argument('--time_dim', type=int, help='Time dimension', default=100)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimension', default=100)
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=5)
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=5)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv




def save_results(new_results: dict, filename: str):
    """
    save (new) results into a json file
    :param: new_results (dictionary): a dictionary of new results to be saved
    :filename: the name of the file to save the (new) results
    """
    if os.path.isfile(filename):
        # append to the file
        with open(filename, 'r+') as json_file:
            file_data = json.load(json_file)
            # convert file_data to list if not
            if type(file_data) is dict:
                file_data = [file_data]
            file_data.append(new_results)
            json_file.seek(0)
            json.dump(file_data, json_file, indent=4)
    else:
        # dump the results
        with open(filename, 'w') as json_file:
            json.dump(new_results, json_file, indent=4)


def split_by_time(data):
    """
    https://github.com/Lee-zix/CEN/blob/main/rgcn/utils.py
    create list where each entry has an entry with all triples for this timestep
    """
    timesteps = list(set(data[:,3]))
    timesteps.sort()
    snapshot_list = [None] * len(timesteps)

    for index, ts in enumerate(timesteps):
        mask = np.where(data[:, 3] == ts)[0]
        snapshot_list[index] = data[mask,:3]

    return snapshot_list



# def read_id_to_string_mappings(root, num_rels):
#     """
#     Reads the node_mapping and rel_mapping csv files and returns the mapping dictionaries.

#     Args:
#         root (str): The directory containing the node_mapping.csv and rel_mapping.csv files.
#             The directory should be the same as the one containing the edgelist.csv file.
    
#     Returns:
#         two dictionaries - node_mapping_dict and rel_mapping_dict.
#             node_mapping_dict (dict): A dictionary mapping node IDs to their corresponding labels and indices.
#             the key is the id as used in the code, the values are a list [string, original_idx as in edgelist.csv]
#             rel_mapping_dict (dict): A dictionary mapping relation IDs to their corresponding labels.
#             the key is the id as used in the code and in edgelist.csv, the value is the string representation of the relation
#     """
#     # load the mapping csv for the node_mapping.csv and the rel_mapping.csv
#     node_mapping = osp.join(root, 'node_mapping.csv')
#     rel_mapping = osp.join(root, 'rel_mapping.csv')

#     node_mapping_dict = {}
#     rel_mapping_dict = {}
#     with open(node_mapping, 'r', encoding='utf-8') as f:
#         for line in f:
#             if not 'original_id' in line:
#                 line = line.strip().split(';') 
#                 try:       
#                     node_mapping_dict[int(line[1])] = [line[2], int(line[0])]
#                 except:
#                     node_mapping_dict[int(line[1])] = ['todo', int(line[0])]
#     with open(rel_mapping, 'r', encoding='utf-8') as f:
#         for line in f:
#             if not 'original_id' in line:
#                 line = line.strip().split(';')        
#                 rel_mapping_dict[int(line[0])] = line[1]
#                 rel_mapping_dict[int(line[0])+num_rels] = 'inv_'+line[1]
#     return node_mapping_dict, rel_mapping_dict



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
                    try:
                        node_mapping_dict[int(line[1])] = [line[2],line[0]]
                    except:
                        node_mapping_dict[int(line[1])] = [line[0],line[0]]
    with open(rel_mapping, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for line in f)
        num_rels = int(num_lines - 1)
        print('num_rels: ', num_rels)
    with open(rel_mapping, 'r', encoding='utf-8') as f:
        for line in f:
            if not 'original_id' in line:
                line = line.strip().split(';')       
                try:

                    rel_mapping_dict[int(line[0])] = line[1]
                    rel_mapping_dict[int(line[0])+num_rels] = 'inv_'+line[1]
                except:
                    try:
                        rel_mapping_dict[int(line[1])] = line[2]
                        rel_mapping_dict[int(line[1])+num_rels] = 'inv_'+line[2]        
                    except:
                        rel_mapping_dict = {}
    return node_mapping_dict, rel_mapping_dict




def fetch_wikidata_property_name(property_id):
    """
    Fetch the name of a Wikidata property given its ID.
    """
    if 'Q' in property_id:
        url = f"https://www.wikidata.org/wiki/{property_id}"
    else:
        url = f"https://www.wikidata.org/wiki/Property:{property_id}"
    response = requests.get(url)

    if response.status_code == 200:
        # Use regular expression to find the title tag content
        match = re.search(r'<title>(.*?) - Wikidata</title>', response.text)
        
        if match:
            property_name = match.group(1).strip()
            return f"{property_name} ({property_id})"
        else:
            return property_id
    else:
        return property_id