import spacy
import dgl
import dgl.frame
import torch
import os, csv
import pandas as pd
from tqdm import tqdm
import benepar
from transformers import AutoTokenizer
import string


class Tree(object):
    def __init__(self, type):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.type = type
        self.is_leaf = False
        self.start = -1
        self.end = -1
        self.idx = -1

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        return count

    def __str__(self):
        return self.type

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def get_cons_tag_vocab(data_path):
    tag2id = {}
    with open(data_path) as f:
        for line in f.readlines():
            tag, idx = line.strip().split('\t')
            tag2id[tag] = int(idx)
    return tag2id


def span_starts_ends(node: Tree):
    if len(node.children) == 0:
        return
    for child in node.children:
        span_starts_ends(child)

    node.start = node.children[0].start
    node.end = node.children[-1].end


def constituent_to_tree(tokenizer, constituent_string, sentence, word_offset, node_offset, num_orders=2):
    constituents = []
    temp_str = ""
    for i, char in enumerate(constituent_string):
        if char == "(" or char == ")" or char == " ":
            if len(temp_str) != 0:
                constituents.append(temp_str)
                temp_str = ""
            if char != " ":
                constituents.append(char)
        else:
            temp_str += char
    # NP, PP등 노드 단위로 stack
    stack = []
    for cons in constituents:
        if cons != ")":
            stack.append(cons)
        else:
            tail = stack.pop()
            temp_constituents = []
            while tail != "(":
                temp_constituents.append(tail)
                tail = stack.pop()

            parent = Tree(temp_constituents[-1])
            for i in range(len(temp_constituents) - 2, -1, -1):
                if isinstance(temp_constituents[i], Tree):
                    parent.add_child(temp_constituents[i])
                else:
                    child = Tree(temp_constituents[i])
                    parent.add_child(child)
            stack.append(parent)
    root = stack[-1]
    map_count = 0
    words = []
    subtokens = []
    subtoken_map = []
    for node in root:
        if len(node.children) == 0:
            node.is_leaf = True
            words.append(str(node))
            node_token = tokenizer.tokenize(str(node))
            if len(node_token) == 0:
                continue
            subtokens.extend(node_token)
            subtoken_map.extend([map_count]*len(node_token))
            map_count += 1

    for node in root:
        if node.is_leaf:
            node.start = word_offset
            node.end = word_offset
            word_offset += 1
    span_starts_ends(root)

    node_sequence = []
    # internal nodes 는 S, NP VP, PP 와같은 노드만. one, lone과같은 노드는 없음
    internal_nodes = []
    for node in root:
        if not node.is_leaf:
            internal_nodes.append(node)
        node_sequence.append(node)

    node_offset_original = node_offset
    for node in root:
        if node.is_leaf:
            continue
        node.idx = node_offset
        node_offset += 1

    constituent_sequence = [] # [(node idx, node start, node end, node type, parent idx)]
    num_internal_nodes = len(internal_nodes)
    # constituent_edge
    constituent_edge = [[0] * num_internal_nodes for _ in range(num_internal_nodes)]
    for i, node in enumerate(internal_nodes):
        parent_idx = node.parent.idx if node.parent else -1
        constituent_sequence.append((node.idx, node.start, node.end, node.type, parent_idx))
        if parent_idx != -1:
            constituent_edge[node.idx - node_offset_original][parent_idx - node_offset_original] = 1 #바로 아래 코드랑 보면 양방향 엣지 포함하는거임
            constituent_edge[parent_idx - node_offset_original][node.idx - node_offset_original] = 1
    # 이부분은 한계층 건너 뛰어서 엣지 이어 주는 식임. 원래 S랑 PP는 안이어져있는데 여기서 이어줌
    high_order_sequence = [constituent_sequence]
    for i in range(1, num_orders):
        new_constituent_sequence = []
        for idx, start, end, type, parent_idx in high_order_sequence[-1]:
            if parent_idx == -1:
                continue
            parent_node = constituent_sequence[parent_idx - node_offset_original]
            if parent_node[-1] == -1:
                continue
            new_constituent_sequence.append((idx, start, end, type, parent_node[-1]))
            constituent_edge[idx - node_offset_original][parent_node[-1] - node_offset_original] = 1
            constituent_edge[parent_node[-1] - node_offset_original][idx - node_offset_original] = 1
        high_order_sequence.append(new_constituent_sequence)
    return high_order_sequence, word_offset, node_offset


def final_graph(constituent_list, first_graph, second_graph):
    cons_tag2id = get_cons_tag_vocab('../../data/IAM/constituent_gold_vocab.txt')
    forward_edge_type, backward_edge_type = 0, 2
    # 여기서 pc, gpc그래프 나눠서 주는게 낫지않을까
    constituent_labels_first = []
    constituent_labels_second = []
    prev_root_node_id = None
    print('fist graph', first_graph.edges())
    print('second graph', second_graph.edges())
    one_order_sent_cons = constituent_list[0][0]
    two_order_sent_cons = constituent_list[0][1]
    for idx, start, end, label, parent_idx in one_order_sent_cons:
        idx_nodeid = idx
        # parent 없는 노드
        if parent_idx == -1:
            if prev_root_node_id is not None:
                first_graph.add_edges(prev_root_node_id, idx_nodeid,
                                data={'cc_link': torch.tensor([1]),
                                      'dtype': torch.tensor([1])})
                # dual GAT
                first_graph.add_edges(idx_nodeid, prev_root_node_id,
                                data={'cc_link': torch.tensor([1]),
                                      'dtype': torch.tensor([1])})
            prev_root_node_id = idx_nodeid
        # parent 없는 노드들
        if parent_idx != -1:
            parent_idx_nodeid = parent_idx
            first_graph.add_edges(parent_idx_nodeid, idx_nodeid,
                            data={'cc_link': torch.tensor([1]),
                                  'dtype': torch.tensor([1])})
            first_graph.add_edges(idx_nodeid, parent_idx_nodeid,
                            data={'cc_link': torch.tensor([1]),
                                  'dtype': torch.tensor([1])})

        # self-loop edge
        first_graph.add_edges(idx_nodeid, idx_nodeid, data={'cc_link': torch.tensor([1]),
                                                      'dtype': torch.tensor([1])})
        constituent_labels_first.append(cons_tag2id[label])
    # print('first graph', first_graph.edges())

    for idx, start, end, label, parent_idx in two_order_sent_cons:
        idx_nodeid = idx
        # parent 없는 노드
        if parent_idx == -1:
            if prev_root_node_id is not None:
                second_graph.add_edges(prev_root_node_id, idx_nodeid,
                                data={'cc_link': torch.tensor([1]),
                                      'dtype': torch.tensor([1])})
                # dual GAT
                second_graph.add_edges(idx_nodeid, prev_root_node_id,
                                data={'cc_link': torch.tensor([1]),
                                      'dtype': torch.tensor([1])})
            prev_root_node_id = idx_nodeid
        # parent 없는 노드들
        if parent_idx != -1:
            parent_idx_nodeid = parent_idx
            second_graph.add_edges(parent_idx_nodeid, idx_nodeid,
                            data={'cc_link': torch.tensor([1]),
                                  'dtype': torch.tensor([1])})
            second_graph.add_edges(idx_nodeid, parent_idx_nodeid,
                            data={'cc_link': torch.tensor([1]),
                                  'dtype': torch.tensor([1])})
        constituent_labels_second.append(cons_tag2id[label])
    print('second graph', second_graph.edges())
    # for high_order_sent_cons in constituent_list:
    #     # i = 0: parent - child/ i = 1: grand parent - grand child
    #     for i, sent_cons in enumerate(high_order_sent_cons):
    #         for idx, start, end, label, parent_idx in sent_cons:
    #             idx_nodeid = idx
    #             # parent 없는 노드
    #             if parent_idx == -1:
    #                 if prev_root_node_id is not None:
    #                     graph.add_edges(prev_root_node_id, idx_nodeid,
    #                                     data={'cc_link': torch.tensor([1]),
    #                                           'dtype': torch.tensor([1])})
    #                     # dual GAT
    #                     graph.add_edges(idx_nodeid, prev_root_node_id,
    #                                     data={'cc_link': torch.tensor([1]),
    #                                           'dtype': torch.tensor([1])})
    #                 prev_root_node_id = idx_nodeid
    #             # parent 없는 노드들
    #             if parent_idx != -1:
    #                 parent_idx_nodeid = parent_idx
    #                 graph.add_edges(parent_idx_nodeid, idx_nodeid,
    #                                 data={'cc_link': torch.tensor([1]),
    #                                       'dtype': torch.tensor([1])})
    #                 graph.add_edges(idx_nodeid, parent_idx_nodeid,
    #                                 data={'cc_link': torch.tensor([1]),
    #                                       'dtype': torch.tensor([1])})
    #
    #             if i == 0:
    #                 # self-loop edge
    #                 graph.add_edges(idx_nodeid, idx_nodeid, data={'cc_link': torch.tensor([1]),
    #                                                               'dtype': torch.tensor([1])})
    #                 constituent_labels.append(cons_tag2id[label])
    #         print(graph.edges(form='all'))

    constituent_labels_first = torch.tensor(constituent_labels_first, dtype=torch.long)
    constituent_labels_second = torch.tensor(constituent_labels_second, dtype=torch.long)
    return first_graph, second_graph, constituent_labels_first, constituent_labels_second


def all_process_graph(nlp, tokenizer, sentence):
    sentence_doc = nlp(sentence)
    sentence_sent = list(sentence_doc.sents)[0]
    parse_string = sentence_sent._.parse_string
    word_offset, node_offset = 0, 0
    constituent = []
    constituent_sequence, word_offset, node_offset = \
        constituent_to_tree(tokenizer, parse_string, sentence, word_offset, node_offset)
    constituent.append(constituent_sequence)

    first_graph = dgl.graph([])
    first_graph.set_n_initializer(dgl.frame.zero_initializer)
    num_cons = sum([len(sent_cons[0]) for sent_cons in constituent])
    first_graph.add_nodes(num_cons)
    first_graph.ndata['unit'] = torch.ones(num_cons)
    first_graph.ndata['dtype'] = torch.ones(num_cons)
    second_graph = dgl.graph([])
    second_graph.set_n_initializer(dgl.frame.zero_initializer)
    num_cons = sum([len(sent_cons[0]) for sent_cons in constituent])
    second_graph.add_nodes(num_cons)
    second_graph.ndata['unit'] = torch.ones(num_cons)
    second_graph.ndata['dtype'] = torch.ones(num_cons)

    claim_first_graph, claim_second_graph, constituent_labels_frist, constituent_labels_second = \
        final_graph(constituent, first_graph, second_graph)
    return claim_first_graph, claim_second_graph, constituent_labels_frist, constituent_labels_second


if __name__ == "__main__":

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    printable = set(string.printable)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=False, use_fast=False)

    train_data = pd.read_csv('../../data/IAM/claims/train.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
    train_data.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
    train_data = train_data.dropna(axis=0)
    dev_data = pd.read_csv('../../data/IAM/claims/dev.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
    dev_data.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
    dev_data = dev_data.dropna(axis=0)

    train_sentences = train_data['claim_sentence'].tolist()[:10]
    dev_sentences = dev_data['claim_sentence'].tolist()[:10]
    total_train = []
    total_dev = []
    for idx, train in tqdm(enumerate(train_sentences), total=len(train_sentences)):
        train = train.lower().replace('“', '"').replace('”', '"')
        train = "".join(filter(lambda x : x in printable, train))

        train_first_graph, train_second_graph, train_constituent_labels_first, train_constituent_labels_second \
            = all_process_graph(nlp, tokenizer, train)
        dgl.save_graphs('../../data/IAM/claims/graphs/train_first_graph_{}.dgl'.format(idx), train_first_graph)
        dgl.save_graphs('../../data/IAM/claims/graphs/train_second_graph_{}.dgl'.format(idx), train_second_graph)
        total_train.append([train_constituent_labels_first.tolist(), train_constituent_labels_second.tolist()])

    for idx, dev in tqdm(enumerate(dev_sentences), total=len(dev_sentences)):
        dev = dev.lower().replace('“', '"').replace('”', '"')
        dev = "".join(filter(lambda x : x in printable, dev))
        dev_first_graph, dev_second_graph, dev_constituent_label_first, dev_constituent_label_second \
            = all_process_graph(nlp, tokenizer, dev)
        dgl.save_graphs('../../data/IAM/claims/graphs/dev_first_graph_{}.dgl'.format(idx), dev_first_graph)
        dgl.save_graphs('../../data/IAM/claims/graphs/dev_second_graph_{}.dgl'.format(idx), dev_second_graph)
        total_dev.append([dev_constituent_label_first.tolist(), dev_constituent_label_second.tolist()])

    with open('../../data/IAM/claims/graphs/train_constituent_test.txt', 'w', encoding='utf-8') as f:
        for line in total_train:
            f.write(str(line)+'\n')

    with open('../../data/IAM/claims/graphs/dev_constituent_test.txt', 'w', encoding='utf-8') as f:
        for line in total_dev:
            f.write(str(line)+'\n')
