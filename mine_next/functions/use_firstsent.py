from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from hdbscan import HDBSCAN
from transformers import BertModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import os, json
from os import listdir
from os.path import isfile, join

def topic_sentences(mode):
    sentences = []
    article_ids = []
    topic_dir_list = os.listdir('../../data/IAM/origin/{}'.format(mode))
    topic_dir_list = [os.path.join('../../data/IAM/origin/{}'.format(mode), topic) for topic in topic_dir_list]

    for topic_dir in topic_dir_list:
        file_list = os.listdir(topic_dir)
        file_list_open = [os.path.join(topic_dir, file) for file in file_list]

        for idx, file in zip(file_list, file_list_open):
            article_id = idx.split('.')[0]
            sentence = []
            with open(file, 'r', encoding='utf-8') as f:
                article = f.readlines()
            for line in article:
                article_sentence = line.split('\t')[0]
                #sentences.append(article_sentence)
                sentence.append(article_sentence)
            sentences.append(sentence[0])
            #sentences.append(' '.join(sent for sent in sentence))
            article_ids.append(article_id)
    return article_ids, sentences

train_ids, train_sentences = topic_sentences('train')
dev_ids, dev_sentences = topic_sentences('dev')
fit_data = train_sentences + dev_sentences
test_ids, test_sentences = topic_sentences('test')
# embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
# # embedding_model = SentenceTransformer("all-mpnet-base-v2")
# cluster_model = HDBSCAN(min_cluster_size=3, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
# topic_model = BERTopic(embedding_model=embedding_model, hdbscan_model=cluster_model)
#topic_model.save('../../data/IAM/origin/topic_model')
#


#topic_model = BERTopic.load('../../data/IAM/origin/topic_model')
#topics, probs = topic_model.fit_transform(fit_data)

#print(topic_model.get_topic_info())

def make_pseudo_topic_with_bertopic(ids, sentences, mode):
    pseudo_topic_dict = {}
    for idx, sentence in zip(ids, sentences):
        #pseudo_topic = topic_model.get_topic(topic=topic_model.transform(sentence)[0][0])
        #pseudo_topic = ' '.join([topic_word[0] for topic_word in pseudo_topic])
        pseudo_topic_dict[idx] = sentence
    with open('../../data/IAM/origin/{}_pseudo_topic_with_first_sent.json'.format(mode), 'w', encoding='utf-8') as file:
        json.dump(pseudo_topic_dict, file, indent='\t', ensure_ascii=False)

make_pseudo_topic_with_bertopic(train_ids, train_sentences, 'train')
make_pseudo_topic_with_bertopic(dev_ids, dev_sentences, 'dev')
make_pseudo_topic_with_bertopic(test_ids, test_sentences, 'test')


# with open('../../data/IAM/origin/test_pseudo_topic_with_bertopic.json', 'w', encoding='utf-8') as file:
#     json.dump(pseudo_topic_dict, file, indent='\t', ensure_ascii=False)