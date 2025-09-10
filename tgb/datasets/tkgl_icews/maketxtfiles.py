import csv

# Input and output file paths

input_file = 'tkgl-icews_edgemapping.csv'
output_file = 'relation2id.txt'


input_file = 'tkgl-icews_nodemapping.csv'
output_file = 'entity2id.txt'



if 'edge' in input_file:
    index_string = 2
    maxnum =3
elif 'node' in input_file:
    index_string = 0
    maxnum = 2
# Read the CSV and write the required columns to the output file
with open(input_file, 'r', encoding='utf-8') as csv_file, open(output_file, 'w', encoding='utf-8') as txt_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        if len(row) >= maxnum:  # Ensure there are at least two columns
            txt_file.write(f"{row[index_string]}\t{row[1]}\n")