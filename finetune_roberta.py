import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

model_output_dir = './model/roberta_finetune'

# 读取数据
with open('./data/multi_nli_train_mini.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]
data = Dataset.from_list(data)

# 过滤无效标签的数据
train_data = [item for item in data if item["label"] != -1]
print(f"after filtering: {len(train_data)}")

# 划分训练集和验证集
train_val_split = data.train_test_split(test_size=0.1, seed=42)
train_data = train_val_split['train']
val_data = train_val_split['test']

# 加载分词器和模型
model_path = './model/roBerta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)

# 预处理函数
def preprocess_function(examples):
    premise = examples["premise"]
    hypothesis = examples["hypothesis"]
    inputs = [f"Premise: {p} Hypothesis: {h}" for p, h in zip(premise, hypothesis)]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    model_inputs["labels"] = examples["label"]
    return model_inputs

# 预处理数据
train_dataset = train_data.map(preprocess_function, batched=True)
val_dataset = val_data.map(preprocess_function, batched=True)

# 训练参数设置
training_args = TrainingArguments(
    output_dir=model_output_dir,
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,
    save_strategy="no"
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 训练模型
trainer.train()

# 保存微调后的模型
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
print("Model config num_labels:", model.config.num_labels)

# 评估模型
results = trainer.evaluate()
print(results)


# 对验证集进行预测
predictions = trainer.predict(val_dataset)

# 提取 logits 并转换为标签
predicted_logits = predictions.predictions
predicted_labels = predicted_logits.argmax(axis=1)  # 获取每个样本的最高概率标签

# 定义标签映射
label_mapping = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

# 输出所有预测标签
print("Predicted labels for validation set:")
for i, label_id in enumerate(predicted_labels):
    print(f"Sample {i + 1}: {label_mapping[label_id]}")