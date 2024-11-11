import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, AlbertTokenizer, AlbertForMaskedLM
import csv
import os
from tqdm import tqdm
import pandas as pd
import difflib

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 




def load_model(lm_model):
    if lm_model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("./model/roBerta-base")
        model = RobertaForMaskedLM.from_pretrained("./model/roBerta-base")
    elif lm_model == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('./model/albert-base-v2')
        model = AlbertForMaskedLM.from_pretrained('./model/albert-base-v2')
    else:
        tokenizer = RobertaTokenizer.from_pretrained("./model/roBerta-base")
        model = RobertaForMaskedLM.from_pretrained("./model/roBerta-base")
    model.eval()
    if torch.cuda.is_available():
        print("cuda is available!")
        model.to('cuda')
    return model ,tokenizer

def get_sentence(data):
    if data['stereo_antistereo'] == 'stereo':
        sent_more = data['sent_more']
        sent_less = data['sent_less']
    else:
        sent_more = data['sent_less']
        sent_less = data['sent_more']
    return sent_more, sent_less

def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, model, tokenizer, log_softmax, mask_token):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]

    target_id = token_ids[0][mask_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs

def get_span(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    """
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2

def mask_unigram(data, model, tokenizer, log_softmax, mask_token, uncased, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    sent_more, sent_less = get_sentence(data)

    if uncased:
        sent_more = sent_more.lower()
        sent_less = sent_less.lower()
    
    # tokenize
    sent_more_token_ids = tokenizer.encode(sent_more, return_tensors='pt')
    sent_less_token_ids = tokenizer.encode(sent_less, return_tensors='pt')

    # get spans of non-changing tokens
    template1, template2 = get_span(sent_more_token_ids[0], sent_less_token_ids[0])

    print(f"check length template {len(template1)}, {len(template2)}")
    assert len(template1) == len(template2)

    N = len(template1) # num. of tokens that can be masked
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    sent_more_log_probs = 0.
    sent_less_log_probs = 0.
    total_masked_tokens = 0

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent_more_masked_token_ids = sent_more_token_ids.clone().detach()
        sent_less_masked_token_ids = sent_less_token_ids.clone().detach()

        sent_more_masked_token_ids[0][template1[i]] = mask_id
        sent_less_masked_token_ids[0][template2[i]] = mask_id
        total_masked_tokens += 1

        score_more = get_log_prob_unigram(sent_more_masked_token_ids, sent_more_token_ids, template1[i], model, tokenizer, log_softmax, mask_token)
        score_less = get_log_prob_unigram(sent_less_masked_token_ids, sent_less_token_ids, template2[i], model, tokenizer, log_softmax, mask_token)

        sent_more_log_probs += score_more.item()
        sent_less_log_probs += score_less.item()

    score = {}
    # average over iterations
    score["sent_more_score"] = sent_more_log_probs
    score["sent_less_score"] = sent_less_log_probs

    return score


def calculate_bias(dataset,lm_model,uncased, output_file):
    """
    Evaluate a masked language model using CrowS-Pairs dataset.
    """
    # load dataset

    # load model
    model, tokenizer = load_model(lm_model)
    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)

    total_stereo, total_antistereo = 0, 0
    stereo_score, antistereo_score = 0, 0
    df_score = pd.DataFrame(columns=['sent_more', 'sent_less', 
                                     'sent_more_score', 'sent_less_score',
                                     'score', 'stereo_antistereo', 'bias_type'])
    N = 0
    neutral = 0
    total = len(dataset.index)
    with tqdm(total=total) as pbar:
        for index, data in dataset.iterrows():
            score = mask_unigram(data,model,tokenizer, log_softmax, mask_token, uncased)

            for stype in score.keys():
                score[stype] = round(score[stype], 3)

            N += 1
            pair_score = 0
            pbar.update(1)

            if score['sent_more_score'] == score['sent_less_score']:
                neutral += 1
            else:
                if data['stereo_antistereo'] == 'stereo':
                    total_stereo += 1
                    if score['sent_more_score'] > score['sent_less_score']:
                        stereo_score += 1
                        pair_score = 1
                elif data['stereo_antistereo'] == 'antistereo':
                    total_antistereo += 1
                    if score['sent_more_score'] > score['sent_less_score']:
                        antistereo_score += 1
                        pair_score = 1
            
            sent_more, sent_less = '', ''
            if data['stereo_antistereo'] == 'stereo':
                sent_more = data['sent_more']
                sent_less = data['sent_less']
                sent_more_score = score['sent_more_score']
                sent_less_score = score['sent_less_score']
            else:
                sent_more = data['sent_less']
                sent_less = data['sent_more']
                sent_more_score = score['sent_less_score']
                sent_less_score = score['sent_more_score']
            new_data = pd.DataFrame([{
                'sent_more': sent_more,
                'sent_less': sent_less,
                'sent_more_score': sent_more_score,
                'sent_less_score': sent_less_score,
                'score': pair_score,
                'stereo_antistereo': data['stereo_antistereo'],
                'bias_type': data['bias_type']
            }])

            df_score = pd.concat([df_score, new_data], ignore_index=True)
    df_score.to_csv(output_file)
    print('Total examples:', N)
    print('Metric score:', round((stereo_score + antistereo_score) / N * 100, 2))
    print('Stereotype score:', round(stereo_score  / total_stereo * 100, 2))
    if antistereo_score != 0:
        print('Anti-stereotype score:', round(antistereo_score  / total_antistereo * 100, 2))
    print("Num. neutral:", neutral, round(neutral / N * 100, 2))
            

def main():
    lm_model = 'albert'
    uncased = True
    # dataset
    dataset = pd.read_csv('./data/bias/sexual_crows_pairs.csv')
    output_file = './data/bias/bias_score_albert.csv'
    print('------------')
    calculate_bias(dataset, lm_model,uncased,output_file)

if __name__ == '__main__':
    main()


"""
roberta result for sexual-oreintation:
Total examples: 84
Metric score: 60.71
Stereotype score: 63.89
Anti-stereotype score: 41.67
Num. neutral: 0 0.0

///
albert result for sexual-orientation:
Total examples: 84
Metric score: 75.0
Stereotype score: 79.17
Anti-stereotype score: 50.0
Num. neutral: 0 0.0
"""
