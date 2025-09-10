import pandas as pd

# Load edgelist.csv and node_mapping.csv
dataset_name = 'tkgl-icews14'
edgelist_df = pd.read_csv(dataset_name+'_edgelist.csv')
node_mapping_df = pd.read_csv('node_mapping.csv', sep=';')

# Create a dictionary for mapping original_id to tgb_id
mapping_dict = dict(zip(node_mapping_df['original_id'], node_mapping_df['tgb_id']))

# Replace 'head' and 'tail' in edgelist_df using the mapping dictionary
edgelist_df['head'] = edgelist_df['head'].map(mapping_dict)
edgelist_df['tail'] = edgelist_df['tail'].map(mapping_dict)

# Save the updated edgelist to a new CSV file
edgelist_df.to_csv(dataset_name+'_edgelist_tgbids.csv', index=False)

print("New edgelist file 'edgelist_new.csv' created successfully.")