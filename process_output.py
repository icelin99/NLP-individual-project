import json
import csv

csv_file = './data/hallucination_output_t5.csv'
output_file = './data/hallucination_output_t5_modified.csv'

# predicted_labels = set()
# # 打开并逐行读取JSONL文件
# with open(jsonl_file, 'r') as file:
#     for line in file:
#         try:
#             data = json.loads(line.strip())
#             if 'predicted_label' in data:
#                 predicted_labels.add(data['predicted_label'])
#         except json.JSONDecodeError:
#             print(f"Error decoding line: {line}")


# print("Unique 'predicted_label' values:")
# for label in predicted_labels:
#     print(label)
# with open(csv_file, 'r', newline='') as file:
#     reader = csv.DictReader(file)  # Assuming the first row contains headers
#     for row in reader:
#         if 'predicted_label' in row:
#             predicted_labels.add(row['predicted_label'])

# # Print unique 'predicted_label' values
# print("Unique 'predicted_label' values:")
# for label in predicted_labels:
#     print(label)
'''
a.
positive
essentially nothing
it is not possible to tell
it's impossible to say
yes
no
false
contract
team were killed
a).
neutral
entailment
contradiction
italic
iterative
'''

label_mapping = {
    'yes': 'entailment',
    'no': 'contradiction',
    'false': 'contradiction',
    'contract': 'contradiction',
    'team were killed': 'neutral',
    'iterative': 'entailment',
    'positive': 'entailment',
    'it\'s impossible to say': 'contradiction',
    'it is not possible to tell': 'contradiction',
    'essentially nothing': 'contradiction',
    'a)': 'neutral',
    'italic': 'neutral'
}

# 存储转换后的predicted_label值
predicted_labels = set()

with open(csv_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames  # Get the header from the original file
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    # Write the header to the output file
    writer.writeheader()

    # Iterate over each row and modify the 'predicted_label'
    for row in reader:
        if 'predicted_label' in row:
            original_label = row['predicted_label']
            
            # Apply label mapping or default to 'neutral'
            if original_label in label_mapping:
                row['predicted_label'] = label_mapping[original_label]
            else:
                row['predicted_label'] = 'neutral'

            # Convert 'predicted_label' to numerical values
            if row['predicted_label'] == 'entailment':
                row['predicted_label'] = 0
            else:
                row['predicted_label'] = 1

        # Write the modified row to the new CSV file
        writer.writerow(row)
# with open(jsonl_file, 'r') as file:
#     for line in file:
#         try:
#             data = json.loads(line.strip())
#             if 'predicted_label' in data:
#                 original_label = data['predicted_label']
#                 # 转换标签
#                 if original_label in label_mapping:
#                     data['predicted_label'] = label_mapping[original_label]
#                 else:
#                     # 默认转换为 neutral
#                     data['predicted_label'] = 'neutral'
                
#                 # 将转换后的predicted_label值加入集合
#                 predicted_labels.add(data['predicted_label'])
#         except json.JSONDecodeError:
#             print(f"Error decoding line: {line}")

# # 输出所有唯一的predicted_label值
# print("Unique 'predicted_label' values after mapping:")
# for label in predicted_labels:
#     print(label)

# # 可选：保存修改后的文件
# with open(output_file, 'w') as outfile:
#     with open(jsonl_file, 'r') as file:
#         for line in file:
#             try:
#                 data = json.loads(line.strip())
#                 if 'predicted_label' in data:
#                     original_label = data['predicted_label']
#                     # mapping label
#                     if original_label in label_mapping:
#                         data['predicted_label'] = label_mapping[original_label]
#                     else:
#                         # convert to neutral
#                         data['predicted_label'] = 'neutral'
#                 # save to new file
#                 outfile.write(json.dumps(data) + '\n')
#             except json.JSONDecodeError:
#                 print(f"Error decoding line: {line}")