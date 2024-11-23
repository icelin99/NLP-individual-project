from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

model_name = 'gpt2-large'  # 模型名称，可以是 gpt2, gpt2-medium, gpt2-large, gpt2-xl 中的任意一个
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)
model.save_pretrained(f"./model/{model_name}")  # 将模型保存到本地
tokenizer.save_pretrained(f"./model/{model_name}")  # 将 tokenizer 保存到本地
