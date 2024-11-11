import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, precision_recall_fscore_support
import csv
import os

csv_file_path = './data/hallucination_output_t5_modified.csv'
output_file = "./data/eval_result.csv"
model = "flan-t5-base"
data_name = "wiki_bio_gpt3_hallucination"
method = "finetune-10000"

results_df = pd.read_csv(csv_file_path)

true_labels = results_df['true_label']
predicted_labels = results_df['predicted_label']

accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)
precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)
# precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="binary")

file_exists = os.path.isfile(output_file)
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["model", "data", "method", "accuracy", "f1", "precision", "recall"])
    # 写入每次评估的结果
    writer.writerow([model, data_name, method, accuracy, f1, precision, recall])

# Accuracy: 0.7421
# Precision: 0.7566
# Recall: 0.9533
# F1 Score: 0.8436