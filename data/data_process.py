import json
import random

file_path = 'data/multi_nli_train.jsonl'


# 原始 jsonl 文件路径
input_file = 'data/multi_nli_train.jsonl'
# 输出的新的 jsonl 文件路径
output_file = 'data/multi_nli_train_mini.jsonl'

# 读取原始 jsonl 文件中的所有行
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()


if len(lines) < 10000:
    raise ValueError("原始文件中的数据条数不足 10,000 条。")

sampled_lines = random.sample(lines, 10000)
print(sampled_lines[0])

with open(output_file, 'w', encoding='utf-8') as f:
    for line in sampled_lines:
        f.write(line)  

