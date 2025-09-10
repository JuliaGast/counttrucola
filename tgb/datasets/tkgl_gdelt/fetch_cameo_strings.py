import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__)))) 
print(sys.path)




def load_cameo_event_codes(file_path):
    """
    Load CAMEO event codes from a lookup file into a dictionary.
    
    :param file_path: Path to the CAMEO event codes file.
    :return: Dictionary mapping event codes to descriptions.
    """
    event_code_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        # reader = csv.reader(file, delimiter='\t')
        for row in file:
            parts = row.strip().split('\t')

            code, description = parts[0], parts[1]
            event_code_dict[code] = description
    return event_code_dict


dataset_name = 'tkgl-gdelt'

dataset_name2 = dataset_name.replace('-', '_')

string_dict ={}
root = osp.dirname(osp.abspath(__file__))
input_dir_rels = osp.join(root, 'relation2id.txt')
cameo_strings = osp.join(root, 'cameo.txt')

event_code_dict = load_cameo_event_codes(cameo_strings)

lines3 =[]
with open(input_dir_rels, 'r', encoding='utf-8') as file:
    for line_r in file:
        parts = line_r.strip().split('\t')
        if len(parts) < 2:
            continue
        string, original_id = parts[0], int(parts[1])

        string2 = event_code_dict[string]

        lines3.append([string, original_id, string2])

output_dir = osp.join(root, 'new_rel2id.txt')

with open(output_dir, 'w', encoding='utf-8') as file:
    for line in lines3:
        string, original_id, string2 = line
        file.write(f"{string}\t{original_id}\t{string2}\n")


print('done')