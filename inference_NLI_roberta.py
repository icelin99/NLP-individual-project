import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
from datasets import Dataset
import json

model_name = "./model/roberta_finetune"  
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

input_file = './data/dev_mismatched_sampled-1.jsonl'
output_file = './data/dev_mismatched_roberta_output.jsonl'

model.eval()

def prompt_format(premise, hypothesis): 
    prompt = f'''Given the following new sentences, predict the relationship between the premise and the hypothesis:
    Premise: {premise}
    Hypothesis: {hypothesis}
    Possible relationships: entailment, contradiction, or neutral.
    Relationship:
    '''
    return prompt

def generate_relationship(premise, hypothesis):
    
    # 使用 tokenizer 进行编码
    inputs = tokenizer(
        f"Premise: {premise} Hypothesis: {hypothesis}",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取 logits 并应用 softmax 来计算概率
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    # 获取最高概率的标签
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    label_mapping = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
    return label_mapping[predicted_class]

with open(input_file, 'r') as f, open(output_file, 'w') as out_f:
    for line in tqdm(f,desc="Predicting", unit="sample"):
        sample = json.loads(line.strip())
        premise = sample['sentence1']
        hypothesis = sample['sentence2']

        relationship = generate_relationship(premise, hypothesis)
        
        print(f"Predicted Relationship: {relationship}")
        result = {
            "pairID": sample["pairID"], 
            "gold_label": sample['gold_label'],  
            "predicted_label": relationship
        }
        # Write the result to the output file as a new line in JSONL format
        out_f.write(json.dumps(result) + '\n')
print(f"Predictions written to {output_file}")