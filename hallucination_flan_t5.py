import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import csv
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

model_name = "./model/t5_finetune"  
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

model.eval()

# load dataset
data_file_path = './data/hallucination/dataset.json'  # 替换为你的本地文件路径
with open(data_file_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
# dataset = dataset[:5]  # 仅使用前100条数据进行测试
output_csv_path = './data/hallucination_output_t5.csv'


label_mapping = {
    "accurate": 0,  # Factual
    "major_inaccurate": 1,  # Non-Factual
    "minor_inaccurate": 1   # Non-Factual
}
label_mapping_nli = {
    "entailment": "entailment",
    "contradiction": "contradiction",
    "neutral": "neutral",
}




def predict_relationship(premise, hypothesis):
    # 构建输入
    inputs = tokenizer(
        f"What is the relationship with premise and hypothesis? Predict only entailment, contradiction or neutral. Premise: {premise} Hypothesis: {hypothesis}. ",
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # 使用模型进行推理
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=10, num_beams=4, early_stopping=True)

    # 获取最高概率的标签索引
    predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()
    # print(f"Generated output tokens: {outputs}")
    print(f"Decoded prediction: {predicted_label}")
    if predicted_label not in label_mapping_nli:
        print(f"Warning: Unrecognized prediction '{predicted_label}', mapping to 'neutral'")
        # return "neutral"
    nli_label_mapping = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }

    # 使用标签映射将索引转换为字符串标签
    # relationship = nli_label_mapping[predicted_class]
    return predicted_label

predictions = []
true_labels = []
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['True Label', 'Predicted Label'])

    for item in tqdm(dataset,desc="Predicting", unit="sample"):
        wiki_text = item["wiki_bio_text"]
        gpt3_sentences = item["gpt3_sentences"]
        annotations = item["annotation"]
        for gpt3_sentence, annotation in zip(gpt3_sentences, annotations):
            true_label = label_mapping.get(annotation)

            # 使用微调后的模型预测
            predicted_relationship = predict_relationship(wiki_text, gpt3_sentence)

            # # 将 NLI 预测映射为幻觉检测标签
            # if predicted_relationship == "entailment":
            #     predicted_label = 0  # Factual
            # else:
            #     predicted_label = 1  # Non-Factual

            # predictions.append(predicted_label)
            predictions.append(predicted_relationship)
            true_labels.append(true_label)
            writer.writerow([true_label, predicted_relationship])



# 计算评价指标
# accuracy = accuracy_score(true_labels, predictions)
# precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")
