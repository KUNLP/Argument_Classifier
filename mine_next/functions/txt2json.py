import csv
import json

import pandas as pd

dataset = pd.read_csv('../../data/IAM/stance/test.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
dataset.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
dataset = dataset.dropna(axis=0)

# claim_sentences = dataset['claim_sentence'].tolist()
# claim_labels = ['non claim' if label is 'O' else 'claim' for label in dataset['claim_label'].tolist()]
stance_sentences = dataset['claim_sentence']
stance_label = dataset['stance_label']

label_dict = {}
label_dict['-1'] = 'contest'
label_dict['1'] = 'support'
label_dict['0'] = 'non-claim'
data_json = []
for sentence, label in zip(stance_sentences, stance_label):
    content = {}
    content['text'] = sentence
    content['label'] = label_dict[str(label)]
    data_json.append(content)

with open('../../data/IAM/stance/IAM_stance_test.json', 'w', encoding='utf-8') as outfile:
    json.dump(data_json, outfile, indent='\t', ensure_ascii=False)