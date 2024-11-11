import json
import os
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

jsonl_file = "./data/roberta-base-finetune/dev_mismatched_roberta_output.jsonl"
output_file = "./data/eval_result.csv"
model = "roberta-base"
data_name = "dev_mismatched_output"
method = "finetune-10000"

gold_labels = []
predicted_labels = []

with open(jsonl_file, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        gold_labels.append(data['gold_label'])
        predicted_labels.append(data['predicted_label'])

accuracy = accuracy_score(gold_labels, predicted_labels)
f1 = f1_score(gold_labels, predicted_labels, average='weighted', zero_division=0)
precision = precision_score(gold_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(gold_labels, predicted_labels, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# save result to csv
file_exists = os.path.isfile(output_file)
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["model", "data", "method", "accuracy", "f1", "precision", "recall"])
    # 写入每次评估的结果
    writer.writerow([model, data_name, method, accuracy, f1, precision, recall])
