import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm

model_path = "./model/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

# Make sure the model is in evaluation mode
model.eval()

# input file
input_file = './data/dev_matched_mini.jsonl'
output_file = './data/dev_matched_mini_gpt2.jsonl'

# set decoding hyperparameters
temperature = 0.7
top_k = 50
top_p = 0.9
max_tokens = 50
batch_size = 10
length_penalty = 1.0

def prompt_format(premise, hypothesis): 
    prompt = f'''Given the following sentences, predict the relationship between the premise and the hypothesis:
    Example 1:
    Premise: "The inscriptions state how he of gentle visage and beloved of the gods, as he described himself, was filled with remorse and converted to the non-violent teachings of Buddha."
    Hypothesis: "He described himself as being gentle and favored by the gods."
    Relationship: entailment

    Example 2:
    Premise: "As long as sufficient skills are retained inhouse to meet the smart buyer approach discussed above, there does not appear to be any greater risk from contracting out a broader range of design review functions, including such services as construction document discipline reviews and code compliance checks, so long as such functions are widely available from a competitive commercial marketplace."
    Hypothesis: "The smart buyer approach needs specific skills such as negotiation."
    Relationship: neutral

    Example 3:
    Premise: "lots of lots of lots of luck on the job market"
    Hypothesis: "Very unlucky on the job market."
    Relationship: contradiction

    Now, given the Premise: {premise}, Hypothesis: {hypothesis}. 
    Predict the relationship between the premise and the hypothesis (choose one from entailment, contradiction, or neutral)
    Relationship: 
    '''
    return prompt

def prompt_format_simple(premise, hypothesis): 
    prompt = f'''Given the premise: {premise}, and hypothesis: {hypothesis}.
    please predict the relationship between the premise and the hypothesis, answer should be short and choose one from entailment, contradiction, or neutral.
    Relationship:
    '''
    return prompt

# inference
def generate_inference(premise, hypothesis):
    prompt = prompt_format(premise, hypothesis)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # inputs = tokenizer.encode_plus(
    #     prompt,
    #     return_tensors="pt",  # 返回 PyTorch 张量
    #     padding=True,          # 填充到最大长度
    #     truncation=True,       # 截断超过最大长度的部分
    #     max_length=512,        # 最大长度，取决于模型的最大 token 数量
    # )
    # input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]

    # generate output
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length = len(input_ids[0]) + max_tokens,
            do_sample=True,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            length_penalty = length_penalty,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            # early_stopping=True
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    relationship = generated_text
    print("output: ",relationship)
    # .split("\n").strip().split()[0]
    
    return relationship

with open(input_file, 'r') as f, open(output_file, 'w') as out_f:
    for line in tqdm(f,desc="Predicting", unit="sample"):
        sample = json.loads(line.strip())
        premise = sample['sentence1']
        hypothesis = sample['sentence2']

        relationship = generate_inference(premise, hypothesis)
        
        print(f"Predicted Relationship: {relationship}")
        result = {
            "pairID": sample["pairID"], 
            "gold_label": sample['gold_label'],  
            "predicted_label": relationship
        }
        # Write the result to the output file as a new line in JSONL format
        out_f.write(json.dumps(result) + '\n')
print(f"Predictions written to {output_file}")