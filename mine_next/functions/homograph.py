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
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)
        #self.cons_type_embeddings = nn.Embedding(82, 300)

    def forward(self, g, h, edge_type=None):
        # Apply graph convolution and activation.
        # cons_node_ids = g.filter(lambda nodes:nodes.data['dtype'] == 1 )
        # cc_edge_id = g.filter(lambda edges : edges.data['dtype'] == edge_type)
        # self_edge_id = g.filter(lambda edges : edges.data['dtype'] == 4)
        # cc_edge_id = torch.cat([cc_edge_id, self_edge_id], dim=0)

        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            # (batch size, 20(아마 히든사이즈))
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        #h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)


class Tree(object):
    def __init__(self, type):
        # self.start, self.end 단어 기준으로 세는것. one 이 0번째, . 이 27번쨰라 root 노드가 start=0, end=27을 가지고 있는거임
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
    # 괄호 ['(', 'S', '(', 'NP', '(', 'NP', '(', 'CD', 'one', ')', '(', 'ADJP', '(', 'RB', 'long', ')' ... ] 이런식으로 (,)나 단어, constituent 단위로 분리
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
    for node in root:
        if len(node.children) == 0:
            node.is_leaf = True

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
            constituent_edge[node.idx - node_offset_original][parent_idx - node_offset_original] = 1 # 바로 아래 코드랑 보면 양방향 엣지 포함하는거임
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

def final_graph(constituent_list, graph):
    cons_tag2id = get_cons_tag_vocab('../../data/IAM/constituent_gold_vocab.txt')
    forward_edge_type, backward_edge_type = 0, 2

    prev_root_node_id = None
    constituent_labels = []
    for high_order_sent_cons in constituent_list:
        for i, sent_cons in enumerate(high_order_sent_cons):
            for idx, start, end, label, parent_idx in sent_cons:
                idx_nodeid = idx  # 원래는 constituent_start_idx = 0, node_id_offset = 406(token id까지였음. 1063중 406이 토큰이고 이후가 node 였었음.)
                # parent 없는 노드
                if parent_idx == -1:
                    if prev_root_node_id is not None:
                        # graph.add_edges(prev_root_node_id, idx_nodeid,
                        #                 data={'cc_link': torch.tensor([forward_edge_type + i]),
                        #                       'dtype': torch.tensor([forward_edge_type + i])})
                        # # dual GAT
                        # graph.add_edges(idx_nodeid, prev_root_node_id,
                        #                 data={'cc_link': torch.tensor([backward_edge_type + i]),
                        #                       'dtype': torch.tensor([backward_edge_type + i])})
                        graph.add_edges(prev_root_node_id, idx_nodeid,
                                        data={'cc_link': torch.tensor([1]),
                                              'dtype': torch.tensor([1])})
                        # dual GAT
                        graph.add_edges(idx_nodeid, prev_root_node_id,
                                        data={'cc_link': torch.tensor([1]),
                                              'dtype': torch.tensor([1])})
                    prev_root_node_id = idx_nodeid
                # parent 있는 노드들
                if parent_idx != -1:
                    parent_idx_nodeid = parent_idx
                    # graph.add_edges(parent_idx_nodeid, idx_nodeid,
                    #                 data={'cc_link': torch.tensor([forward_edge_type + i]),
                    #                       'dtype': torch.tensor([forward_edge_type + i])})
                    # graph.add_edges(idx_nodeid, parent_idx_nodeid,
                    #                 data={'cc_link': torch.tensor([backward_edge_type + i]),
                    #                       'dtype': torch.tensor([backward_edge_type + i])})
                    graph.add_edges(parent_idx_nodeid, idx_nodeid,
                                    data={'cc_link': torch.tensor([1]),
                                          'dtype': torch.tensor([1])})
                    graph.add_edges(idx_nodeid, parent_idx_nodeid,
                                    data={'cc_link': torch.tensor([1]),
                                          'dtype': torch.tensor([1])})
                if i == 0:
                    # self-loop edge
                    # graph.add_edges(idx_nodeid, idx_nodeid, data={'cc_link': torch.tensor([4]),
                    #                                               'dtype': torch.tensor([4])})
                    graph.add_edges(idx_nodeid, idx_nodeid, data={'cc_link': torch.tensor([1]),
                                                                  'dtype': torch.tensor([1])})
                    constituent_labels.append(cons_tag2id[label])

    constituent_labels = torch.tensor(constituent_labels,dtype=torch.long)
    return graph, constituent_labels

def all_process_graph(nlp, tokenizer, sentence):
    sentence_doc = nlp(sentence)
    sentence_sent = list(sentence_doc.sents)[0]
    parse_string = sentence_sent._.parse_string
    word_offset, node_offset = 0, 0
    constituent = []
    constituent_sequence, word_offset, node_offset = \
        constituent_to_tree(tokenizer, parse_string, sentence, word_offset, node_offset)
    constituent.append(constituent_sequence)

    graph = dgl.graph([])
    graph.set_n_initializer(dgl.frame.zero_initializer)
    num_cons = sum([len(sent_cons[0]) for sent_cons in constituent])

    graph.add_nodes(num_cons)
    graph.ndata['unit'] = torch.ones(num_cons)
    graph.ndata['dtype'] = torch.ones(num_cons)

    claim_graph, constituent_labels = \
        final_graph(constituent, graph)
    return claim_graph, constituent_labels


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=False, use_fast=False)

    dev_data = pd.read_csv('../../data/IAM/claims/dev.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
    dev_data.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
    dev_data = dev_data.dropna(axis=0)

    dev_sentences = dev_data['claim_sentence'].tolist()[:10]
    total_dev_constituent_label = []
    printable = set(string.printable)
    total_graph = {}
    cons_type_embeddings = nn.Embedding(82, 300)
    model = Classifier(300, 300, 2) # homo graph 테스트용

    for idx, dev in tqdm(enumerate(dev_sentences), total=len(dev_sentences)):
        dev = dev.lower().replace('“', '"').replace('”', '"')
        dev = "".join(filter(lambda x: x in printable, dev))

        dev_graph, dev_constituent_label = all_process_graph(nlp, tokenizer, dev)
        total_dev_constituent_label.append([dev_constituent_label])
        total_graph[idx] = dev_graph
        cons_node_feat = cons_type_embeddings(dev_constituent_label)
        #etypes = ['0', '1', '2', '3', '4']
        #model = HeteroClassifier(300, 300, 2, etypes)
        #print(dev_graph.edges(form='all'))
        logits = model(dev_graph, cons_node_feat) # homo
        #logits = model(dev_graph, cons_node_feat) # hetero
        print(logits)


