from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import json
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

# load train data
data_file_path = 'data/multi_nli_train_mini.jsonl'
with open(data_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
dataset = [json.loads(line) for line in lines]
dataset = Dataset.from_list(dataset)

# filtering useless label
train_dataset = [item for item in dataset if item["label"] != -1]
print(f"after filtering: {len(train_dataset)}")

# split 20% for valiadation dataset
train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']


# load model
model_name = './model/flan-t5-base' 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

model_output_path = "./model/t5_finetune"

# prepare input data
def preprocess_function(examples):
    inputs = [f"Premise: {x} Hypothesis: {y} What is the relationship?" for x, y in zip(examples["premise"], examples["hypothesis"])]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True)
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    labels = [label_map[label] for label in examples["label"]]
    labels = tokenizer(labels, padding="max_length", truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs


train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = val_dataset.map(preprocess_function, batched=True)

# setup traning parameter
training_args = TrainingArguments(
    output_dir=model_output_path,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    save_strategy="no"
)

# finetune
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print(train_dataset[0])  # 检查预处理输出样本的格式是否正确
trainer.train()

# 强制保存模型
model.save_pretrained(model_output_path)
tokenizer.save_pretrained(model_output_path)

# 评估
results = trainer.evaluate()
print(results)
