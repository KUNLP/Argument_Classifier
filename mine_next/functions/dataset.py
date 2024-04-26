import json, ast
import benepar
import dgl.frame
from torch.utils.data import TensorDataset, Dataset
import torch
from transformers import AutoTokenizer
import pandas as pd
import argparse
from tqdm import tqdm
import spacy
from mine_next.functions.sent_to_graph import constituent_to_tree, get_cons_tag_vocab, final_graph, all_process_graph
import os, string

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})


def convert_only_sentence2tensordataset(dataset, pseudo, tokenizer, max_length, mode):
    printable = set(string.printable)
    total_idx = []
    total_input_ids = []
    total_attention_mask = []
    total_label = []
    total_token_type_ids = []
    total_sim_label = []
    claim_sentences = dataset['claim_sentence'].tolist()
    claim_labels = dataset['claim_label'].tolist()
    claim_article_id = dataset['article_id'].tolist()
    gold_topic_sentences = dataset['topic_sentence'].tolist()

    claim_labels = [0 if label is 'O' else 1 for label in claim_labels]
    # 여기부분은 평소에 불러다 쓸때 사용하는 부분
    # total_graph = {}
    # max_constituent_length = 600
    # total_constituent_labels = []
    # with open('../data/IAM/claims/graphs/{}_constituent.txt'.format(mode), 'r', encoding='utf-8') as f:
    #     constituents = f.readlines()
    # # 테스트
    # for constituent in constituents:
    #     constituent = ast.literal_eval(constituent.replace('\n', ''))
    #     total_constituent_labels.append(constituent+[-1]*(max_constituent_length-len(constituent)))
    # graphs = os.listdir('../data/IAM/claims/graphs')
    # graphs_list = [os.path.join('../data/IAM/claims/graphs', file) for file in graphs if ('dgl' in file and mode in file)] #train이나 dev 의 dgl만
    # for graph in graphs_list:
    #     (g,), _ = dgl.load_graphs(graph)
    #     idx = graph.split('/')[-1].split('_')[-1].split('.')[0]
    #     total_graph[int(idx)] = g

    # 평소 불러쓸때
    total_graph_first = {}
    total_graph_second = {}
    max_constituent_length = 600
    total_constituent_label_first = []
    total_constituent_label_second = []
    with open('../data/IAM/claims/graphs/{}_constituent_first_second.txt'.format(mode), 'r', encoding='utf-8') as f:
        constituents = f.readlines()
    for constituent in constituents:
        constituent = ast.literal_eval(constituent.replace('\n', ''))
        total_constituent_label_first.append(constituent[0]+[-1]*(max_constituent_length-len(constituent[0])))
        total_constituent_label_second.append(constituent[1]+[-1]*(max_constituent_length-len(constituent[1])))
    graphs = os.listdir('../data/IAM/claims/graphs')
    graphs_list = [os.path.join('../data/IAM/claims/graphs', file) for file in graphs if ('dgl' in file and mode in file and 'first' in file)] #train이나 dev 의 dgl만
    for graph in graphs_list:
        (g,), _ = dgl.load_graphs(graph)
        idx = graph.split('/')[-1].split('_')[-1].split('.')[0]
        total_graph_first[int(idx)] = g
    graphs_list = [os.path.join('../data/IAM/claims/graphs', file) for file in graphs if ('dgl' in file and mode in file and 'second' in file)] #train이나 dev 의 dgl만
    for graph in graphs_list:
        (g,), _ = dgl.load_graphs(graph)
        idx = graph.split('/')[-1].split('_')[-1].split('.')[0]
        total_graph_second[int(idx)] = g

    for idx, (topic, claim_sentence, claim_label, article_id) in tqdm(enumerate(zip(gold_topic_sentences, claim_sentences, claim_labels, claim_article_id)), desc='convert to data to tensordataset', total=len(claim_labels)):
        claim_sentence = claim_sentence.lower().replace('“', '"').replace('”', '"')
        claim_sentence = "".join(filter(lambda x : x in printable, claim_sentence))

        # claim_graph_first, claim_graph_second, constituent_label_first, constituent_label_second = all_process_graph(nlp, tokenizer, claim_sentence)
        # total_graph_first[idx] = claim_graph_first
        # total_graph_second[idx] = claim_graph_second
        # constituent_label_first = constituent_label_first.tolist() + [-1]*(max_constituent_length-len(constituent_label_first.tolist()))
        # constituent_label_second = constituent_label_second.tolist() + [-1]*(max_constituent_length-len(constituent_label_second.tolist()))
        # total_constituent_label_first.append(constituent_label_first)
        # total_constituent_label_second.append(constituent_label_second)

        # 슈도 토픽 할때
        #process_sentence = tokenizer(pseudo[article_id], claim_sentence, max_length=max_length, padding='max_length', truncation=True)
        #process_sentence = tokenizer(claim_sentence, max_length=max_length, padding='max_length', truncation=True)
        process_sentence = tokenizer(topic, claim_sentence, max_length=max_length, padding='max_length', truncation=True)
        input_ids = process_sentence['input_ids']
        attention_mask = process_sentence['attention_mask']
        # 주제에 대한부분만 1로 하고 나머진 0 으로 하는 식
        sep_index = [idx for idx, ids in enumerate(input_ids) if ids == 2]
        try:
            second_sep_index = sep_index[1]
            token_type_ids = [0] * second_sep_index
            token_type_ids += [1] * (len(input_ids)-len(token_type_ids))
        except IndexError:
            token_type_ids = [0] * max_length
        # 주장일 때
        if claim_label == 1:
            sim_label = 1
        # 주장이 아닐때
        elif claim_label == 0:
            sim_label = -1

        total_idx.append(idx)
        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)
        total_label.append(claim_label)
        total_sim_label.append(sim_label)
        #total_graph[idx] = claim_graph
        #total_constituent_labels.append(constituent_label_list)
        if idx < 3:
            print()
            print("****EXAMPLE****")
            print("topic sentence : {}".format(topic))
            print("claim sentence : {}".format(claim_sentence))
            print("claim sentence input ids : {}".format(input_ids))
            print("claim sentence attention mask : {}".format(attention_mask))
            print("claim sentence token type ids : {}".format(token_type_ids))
            print("label : {}".format(claim_label))
            print("sim label : {}".format(sim_label))


    total_idx = torch.tensor(total_idx, dtype=torch.long)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
    total_label = torch.tensor(total_label, dtype=torch.long)
    total_sim_label = torch.tensor(total_sim_label, dtype=torch.long)
    total_constituent_label_first = torch.tensor(total_constituent_label_first, dtype=torch.long)
    total_constituent_label_second = torch.tensor(total_constituent_label_second, dtype=torch.long)
    dataset = TensorDataset(total_idx, total_input_ids, total_attention_mask, total_token_type_ids, total_label, total_sim_label,
                            total_constituent_label_first, total_constituent_label_second)

    return dataset, total_graph_first, total_graph_second


def convert_only_sentence2tensordataset(dataset, pseudo, tokenizer, max_length, mode):
    printable = set(string.printable)
    total_idx = []
    total_input_ids = []
    total_attention_mask = []
    total_label = []
    total_token_type_ids = []
    total_sim_label = []
    claim_sentences = dataset['claim_sentence'].tolist()
    claim_labels = dataset['claim_label'].tolist()
    claim_article_id = dataset['article_id'].tolist()
    gold_topic_sentences = dataset['topic_sentence'].tolist()

    claim_labels = [0 if label is 'O' else 1 for label in claim_labels]
    # 여기부분은 평소에 불러다 쓸때 사용하는 부분
    # total_graph = {}
    # max_constituent_length = 600
    # total_constituent_labels = []
    # with open('../data/IAM/claims/graphs/{}_constituent.txt'.format(mode), 'r', encoding='utf-8') as f:
    #     constituents = f.readlines()
    # #테스트
    # for constituent in constituents:
    #     constituent = ast.literal_eval(constituent.replace('\n', ''))
    #     total_constituent_labels.append(constituent+[-1]*(max_constituent_length-len(constituent)))
    # graphs = os.listdir('../data/IAM/claims/graphs')
    # graphs_list = [os.path.join('../data/IAM/claims/graphs', file) for file in graphs if ('dgl' in file and mode in file)] #train이나 dev 의 dgl만
    # for graph in graphs_list:
    #     (g,), _ = dgl.load_graphs(graph)
    #     idx = graph.split('/')[-1].split('_')[-1].split('.')[0]
    #     total_graph[int(idx)] = g

    #평소 불러쓸때
    total_graph_first = {}
    total_graph_second = {}
    max_constituent_length = 600
    total_constituent_label_first = []
    total_constituent_label_second = []
    with open('../data/IAM/claims/graphs/{}_constituent_first_second.txt'.format(mode), 'r', encoding='utf-8') as f:
        constituents = f.readlines()
    for constituent in constituents:
        constituent = ast.literal_eval(constituent.replace('\n', ''))
        total_constituent_label_first.append(constituent[0]+[-1]*(max_constituent_length-len(constituent[0])))
        total_constituent_label_second.append(constituent[1]+[-1]*(max_constituent_length-len(constituent[1])))
    graphs = os.listdir('../data/IAM/claims/graphs')
    graphs_list = [os.path.join('../data/IAM/claims/graphs', file) for file in graphs if ('dgl' in file and mode in file and 'first' in file)] #train이나 dev 의 dgl만
    for graph in graphs_list:
        (g,), _ = dgl.load_graphs(graph)
        idx = graph.split('/')[-1].split('_')[-1].split('.')[0]
        total_graph_first[int(idx)] = g
    graphs_list = [os.path.join('../data/IAM/claims/graphs', file) for file in graphs if ('dgl' in file and mode in file and 'second' in file)] #train이나 dev 의 dgl만
    for graph in graphs_list:
        (g,), _ = dgl.load_graphs(graph)
        idx = graph.split('/')[-1].split('_')[-1].split('.')[0]
        total_graph_second[int(idx)] = g

    for idx, (topic, claim_sentence, claim_label, article_id) in tqdm(enumerate(zip(gold_topic_sentences, claim_sentences, claim_labels, claim_article_id)), desc='convert to data to tensordataset', total=len(claim_labels)):
        claim_sentence = claim_sentence.lower().replace('“', '"').replace('”', '"')
        claim_sentence = "".join(filter(lambda x : x in printable, claim_sentence))

        # claim_graph_first, claim_graph_second, constituent_label_first, constituent_label_second = all_process_graph(nlp, tokenizer, claim_sentence)
        # total_graph_first[idx] = claim_graph_first
        # total_graph_second[idx] = claim_graph_second
        # constituent_label_first = constituent_label_first.tolist() + [-1]*(max_constituent_length-len(constituent_label_first.tolist()))
        # constituent_label_second = constituent_label_second.tolist() + [-1]*(max_constituent_length-len(constituent_label_second.tolist()))
        # total_constituent_label_first.append(constituent_label_first)
        # total_constituent_label_second.append(constituent_label_second)

        # 슈도 토픽 할때
        process_sentence = tokenizer(pseudo[article_id], claim_sentence, max_length=max_length, padding='max_length', truncation=True)
        # 그냥 문장 하나만 할 때
        # process_sentence = tokenizer(claim_sentence, max_length=max_length, padding='max_length', truncation=True)
        # 골든 토픽과 문장 하나
        # process_sentence = tokenizer(topic, claim_sentence, max_length=max_length, padding='max_length', truncation=True)

        input_ids = process_sentence['input_ids']
        attention_mask = process_sentence['attention_mask']
        # 주제에 대한부분만 1로 하고 나머진 0 으로 하는 식
        sep_index = [idx for idx, ids in enumerate(input_ids) if ids == 2]
        try:
            second_sep_index = sep_index[1]
            token_type_ids = [0] * second_sep_index
            token_type_ids += [1] * (len(input_ids)-len(token_type_ids))
        except IndexError:
            token_type_ids = [0] * max_length
        # 주장일 때
        if claim_label == 1:
            sim_label = 1
        # 주장이 아닐때
        elif claim_label == 0:
            sim_label = -1

        total_idx.append(idx)
        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)
        total_label.append(claim_label)
        total_sim_label.append(sim_label)
        #total_graph[idx] = claim_graph
        #total_constituent_labels.append(constituent_label_list)
        if idx < 3:
            print()
            print("****EXAMPLE****")
            print("topic sentence : {}".format(topic))
            print("pseudo topic sentence : {}".format(pseudo[article_id]))
            print("claim sentence : {}".format(claim_sentence))
            print("claim sentence input ids : {}".format(input_ids))
            print("claim sentence attention mask : {}".format(attention_mask))
            print("claim sentence token type ids : {}".format(token_type_ids))
            print("label : {}".format(claim_label))
            print("sim label : {}".format(sim_label))


    total_idx = torch.tensor(total_idx, dtype=torch.long)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
    total_label = torch.tensor(total_label, dtype=torch.long)
    total_sim_label = torch.tensor(total_sim_label, dtype=torch.long)
    total_constituent_label_first = torch.tensor(total_constituent_label_first, dtype=torch.long)
    total_constituent_label_second = torch.tensor(total_constituent_label_second, dtype=torch.long)
    dataset = TensorDataset(total_idx, total_input_ids, total_attention_mask, total_token_type_ids, total_label, total_sim_label,
                            total_constituent_label_first, total_constituent_label_second)

    return dataset, total_graph_first, total_graph_second


def convert_data2tensordataset(dataset, tokenizer, max_length, mode):
    total_input_ids = []
    total_attention_mask = []
    total_label = []
    total_token_type_ids = []
    total_sim_label = []
    total_idx = []
    claim_sentences = dataset['claim_sentence'].tolist()
    claim_labels = dataset['claim_label'].tolist()
    claim_labels = [0 if label is 'O' else 1 for label in claim_labels]
    topic_sentences = dataset['topic_sentence'].tolist()
    for idx, (topic_sentence, claim_sentence, claim_label) in tqdm(enumerate(zip(topic_sentences, claim_sentences, claim_labels)), desc='convert to data to tensordataset', total=len(claim_labels)):
        process_sentence = tokenizer(topic_sentence, claim_sentence, max_length=max_length, padding='max_length', truncation=True)
        input_ids = process_sentence['input_ids']
        attention_mask = process_sentence['attention_mask']
        # 주제에 대한부분만 1로 하고 나머진 0 으로 하는 식
        sep_index = [idx for idx, ids in enumerate(input_ids) if ids == 2]
        second_sep_index = sep_index[1]
        token_type_ids = [0] * second_sep_index
        token_type_ids += [1] * (len(input_ids)-len(token_type_ids))
        # 주장일 때
        if claim_label == 1:
            sim_label = 1
        # 주장이 아닐때
        elif claim_label == 0:
            sim_label = -1
        total_idx.append(idx)
        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)
        total_label.append(claim_label)
        total_sim_label.append(sim_label)
        if idx < 3:
            print()
            print("****EXAMPLE****")
            print("topic sentence : {}".format(topic_sentence))
            print("claim sentence : {}".format(claim_sentence))
            print("topic, claim sentence input ids : {}".format(input_ids))
            print("topic, claim sentence attention mask : {}".format(attention_mask))
            print("topic, claim sentence token type ids : {}".format(token_type_ids))
            print("label : {}".format(claim_label))
            print("sim label : {}".format(sim_label))
    total_idx = torch.tensor(total_idx, dtype=torch.long)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
    total_label = torch.tensor(total_label, dtype=torch.long)
    total_sim_label = torch.tensor(total_sim_label, dtype=torch.long)
    dataset = TensorDataset(total_idx, total_input_ids, total_attention_mask, total_token_type_ids, total_label, total_sim_label)
    return dataset


def convert_stance_data2tensordataset(dataset, tokenizer, max_length, mode=None):
    total_idx = []
    total_input_ids = []
    total_attention_mask = []
    total_label = []
    total_token_type_ids = []
    total_sim_label = []
    total_stance_label = []
    #dataset = dataset[dataset['claim_label'] == 'C']

    claim_sentences = dataset['claim_sentence'].tolist()
    topic_sentences = dataset['topic_sentence'].tolist()
    stance_labels = dataset['stance_labels'].tolist()
    for idx, (topic_sentence, claim_sentence, stance_label) in tqdm(enumerate(zip(topic_sentences, claim_sentences, stance_labels)), desc='convert to data to tensordataset', total=len(stance_labels)):
        process_sentence = tokenizer(topic_sentence, claim_sentence, max_length=max_length, padding='max_length', truncation=True)
        input_ids = process_sentence['input_ids']
        attention_mask = process_sentence['attention_mask']
        # 주제에 대한부분만 1로 하고 나머진 0 으로 하는 식
        try:
            sep_index = [idx for idx, ids in enumerate(input_ids) if ids == 2]
            second_sep_index = sep_index[1]
            token_type_ids = [0] * second_sep_index
            token_type_ids += [1] * (len(input_ids)-len(token_type_ids))
        except IndexError:
            token_type_ids = [0] * max_length
        #sent_attention_mask = (1-token_type_ids) * attention_mask
        total_idx.append(idx)
        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)
        if stance_label == -1:
            total_stance_label.append(0)
        else:
            total_stance_label.append(1)
        #total_stance_label.append(stance_label)
        if idx < 3:
            print()
            print("****EXAMPLE****")
            print("topic sentence : {}".format(topic_sentence))
            print("claim sentence : {}".format(claim_sentence))
            print("topic, claim sentence input ids : {}".format(input_ids))
            print("topic, claim sentence attention mask : {}".format(attention_mask))
            print("topic, claim sentence token type ids : {}".format(token_type_ids))
            print("stance label : {}".format(stance_label))

    total_idx = torch.tensor(total_idx, dtype=torch.long)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
    total_stance_label = torch.tensor(total_stance_label, dtype=torch.long)
    dataset = TensorDataset(total_idx, total_input_ids, total_attention_mask, total_token_type_ids, total_stance_label)
    return dataset

# with open('../../../data/train_claim.json', 'r', encoding='utf-8') as reader:
#     dataset = json.load(reader)['data']
#
#     total_title = []
#     total_input_ids = []
#     total_attention_mask = []
#     total_label = []
#     for data in dataset:
#         title = data['title']
#         total_title.append(title)
#         paragraphs = data['paragraphs']
#         for para in paragraphs:
#             answers = para['qas'][0]['answers']
#             context = para['context']
#             result = tokenizer(context, padding='max_length', max_length=4096, truncation=True)
#             # cls idx 2 / sep idx 3
#             total_input_ids.append(result['input_ids'])
#             total_attention_mask.append(result['attention_mask'])
#             context_list = context.split('[SEP]')
#             each_label = [0] * len(context_list)
#             # 첫 sep는 첫번째 문장에 대한 표현. 문장의 오른쪽에 있는 sep를 기준으로 한다.
#             for answer in answers:
#                 text = answer['text']
#                 for idx, ctx in enumerate(context_list):
#                     if text in ctx:
#                         print(idx+1)



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='dataset creating')
#     parser.add_argument('--train_data', type=str, default='../../../data/train_claim.json')
