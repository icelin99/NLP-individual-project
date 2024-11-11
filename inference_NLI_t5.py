import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import torch

# load model
model_path = "./model/t5_finetune"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

label_mapping = {
    "entailment": "entailment",
    "contradiction": "contradiction",
    "neutral": "neutral",
}

def predict(model, tokenizer, premise, hypothesis):
    input_text = f"Premise: {premise} Hypothesis: {hypothesis} What is the relationship? Answer the label only, entailment, contradiction or neutral."
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=10, num_beams=4, early_stopping=True)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()
    if prediction not in label_mapping:
        print(f"Warning: Unrecognized prediction '{prediction}', mapping to 'neutral'")
        # return "neutral"
    return prediction
    
def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def save_predictions(predictions, output_file):
    with open(output_file, "w") as f:
        for item in predictions:
            f.write(json.dumps(item) + "\n")

def run_predictions(input_file):
    data = read_jsonl(input_file)
    
    predictions = []
    for entry in tqdm(data,desc="Predicting", unit="sample"):
        premise = entry['sentence1']
        hypothesis = entry['sentence2']
        
        predicted_label = predict(model, tokenizer, premise, hypothesis)
        
        predictions.append({
            "pairID": entry["pairID"],
            "gold_label": entry["gold_label"],
            "predicted_label": predicted_label
        })
    
    return predictions

input_file = "./data/dev_mismatched_sampled-1.jsonl"
output_file = "./data/flan-t5-base/dev_mismatched_output.jsonl"

predictions = run_predictions(input_file)

save_predictions(predictions, output_file)

print(f"prediction result saved: {output_file}")