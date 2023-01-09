import pandas as pd
import csv
from transformers import RobertaTokenizer
import string

s = "“Let’s say there’s a government-run test."
printable = set(string.printable)
print(printable)
print("".join(filter(lambda x: x in printable, s)))


all_claim = []

def extract_claim(data_file):
    data = pd.read_csv(data_file, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    data.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
    data = data.dropna(axis=0)
    data = data[data['claim_label'] == 'C']
    claim_data = data['claim_sentence']
    return claim_data.tolist()
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# input_str = 'a significant number of republicans assert that hereditary monarchy is unfair and elitist'
#print(tokenizer.tokenize(input_str))

# train_claim = extract_claim('../../data/IAM/claims/train.txt')
# print(train_claim)
# dev_claim = extract_claim('../../data/IAM/claims/dev.txt')
# test_claim = extract_claim('../../data/IAM/claims/test.txt')
#
# all_claim.extend(train_claim)
# all_claim.extend(dev_claim)
# all_claim.extend(test_claim)
#
# with open('../../data/IAM/all_claim_sentence.txt', 'w', encoding='utf-8') as txt_file:
#     for claim in all_claim:
#         txt_file.write(claim)
#         txt_file.write('\n')

