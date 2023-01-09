import spacy
import pytextrank
import os
import pandas as pd
import csv
import json, string
from tqdm import tqdm

def make_article_dict():
    topic_dir_list = os.listdir('../../data/IAM/origin/test')
    topic_dir_list = [os.path.join('../../data/IAM/origin/test', topic) for topic in topic_dir_list]
    article_dict = {}
    for topic_dir in topic_dir_list:
        file_list = os.listdir(topic_dir) # [21_3.txt, 21_7.txt, ... ]
        file_list_open = [os.path.join(topic_dir, file) for file in file_list] # ../../data/IAM/origin/train/Should_commercial_advertisements_be_allowed_to_be_fictitious/21_3.txt

        for idx, file in zip(file_list, file_list_open):
            article_id = idx.split('.')[0]

            num = file.split('/')[-1]
            assert idx == num
            sentences = []
            with open(file, 'r', encoding='utf-8') as f:
                article = f.readlines()
            for line in article:
                article_sentence = line.split('\t')[0]
                sentences.append(article_sentence)
            article_dict[article_id] = sentences

    with open('../../data/IAM/origin/test_article_dict.json', 'w', encoding='utf-8') as outfile:
        json.dump(article_dict, outfile, indent='\t', ensure_ascii=False)


def make_pseudo_topic_with_textrank():
    printable = set(string.printable)
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    datas = json.load(open('../../data/IAM/origin/dev_article_dict.json', encoding='utf-8'))
    pseudo_topic = {}
    for key, value in tqdm(datas.items(), total=len(datas)):
        article_text = " ".join(value)
        article_text = article_text.lower().replace('“', '"').replace('”', '"')
        article_text = "".join(filter(lambda x : x in printable, article_text))
        doc = nlp(article_text)
        topic = []
        for phrase in doc._.phrases[:10]:
            topic.append(phrase.text)
        # pseudo_topic[key] = " ".join(topic)
        pseudo_topic[key] = topic
    with open('../../data/IAM/origin/dev_pseudo_topic_with_textrank_list.json', 'w', encoding='utf-8') as file:
        json.dump(pseudo_topic, file, indent='\t', ensure_ascii=False)

#make_article_dict()
# make_pseudo_topic_with_textrank()
# data = json.load(open('../../data/IAM/origin/dev_pseudo_topic_with_textrank_list.json', encoding='utf-8'))
# print(data)


# doc = nlp(article.lower())
#
# # examine the top-ranked phrases in the document
# pseudo_topic = []
# for phrase in doc._.phrases[:10]:
#     #print(phrase)
#     print(phrase.text)
#     # print(phrase.rank, phrase.count)
#     # print(phrase.chunks)
#     print()
#


total_char_count = 0
total_word_count = 0
topic_dir_list = os.listdir('../../data/IAM/origin/test')
topic_dir_list = [os.path.join('../../data/IAM/origin/test', topic) for topic in topic_dir_list]
article_dict = {}
for topic_dir in topic_dir_list:
    file_list = os.listdir(topic_dir) # [21_3.txt, 21_7.txt, ... ]
    file_list_open = [os.path.join(topic_dir, file) for file in file_list] # ../../data/IAM/origin/train/Should_commercial_advertisements_be_allowed_to_be_fictitious/21_3.txt

    for idx, file in zip(file_list, file_list_open):
        article_id = idx.split('.')[0]

        num = file.split('/')[-1]
        assert idx == num
        sentences = []
        with open(file, 'r', encoding='utf-8') as f:
            article = f.readlines()
        for line in article:
            article_sentence = line.split('\t')[0]
            word_of_sentence = article_sentence.split(' ')
            total_char_count += len(article_sentence)
            total_word_count += len(word_of_sentence)
print(total_char_count)
print(total_word_count)