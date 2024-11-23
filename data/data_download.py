from datasets import load_dataset
import json

ds = load_dataset("nyu-mll/multi_nli")

# save to jsonl 
def save_to_jsonl(dataset, filename):
    with open(filename, 'w') as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write("\n")


save_to_jsonl(ds['train'], 'multi_nli_train.jsonl')