import benepar, spacy
from nltk.tree import Tree as nltk_tree
from nltk.treeprettyprinter import TreePrettyPrinter
from nltk.draw.tree import TreeView
import os, csv
import pandas as pd
from tqdm import tqdm
import dgl
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
data = pd.read_csv('../../data/IAM/claims/train.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
data.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
data = data.dropna(axis=0)
data = data[data['claim_label'] == 'C']
claims = data['claim_sentence'].tolist()

def get_cons_tag_vocab(data_path):
    tag2id = {}
    with open(data_path) as f:
        for line in f.readlines():
            tag, idx = line.strip().split('\t')
            tag2id[tag] = int(idx)
    return tag2id



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

def span_starts_ends(node: Tree):
    if len(node.children) == 0:
        return
    for child in node.children:
        span_starts_ends(child)

    node.start = node.children[0].start
    node.end = node.children[-1].end

def constituent_to_tree(constituent_string, word_offset, node_offset, num_orders=2):
    constituents = []
    temp_str = ""
    words = []
    subtokens = []
    subtoken_map = []
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
                    # parent에 붙일때 parent의 leaf를 true로 바꿔주는 형식으로 해줄것
                    child = Tree(temp_constituents[i])
                    parent.add_child(child)
            stack.append(parent)
    root = stack[-1]
    # 노드 방문하면서 잎인지 체크해야함
    map_count = 0
    for node in root:
        if len(node.children) == 0:
            node.is_leaf = True
            words.append(str(node))
            node_token = tokenizer.tokenize(str(node))
            subtokens.extend(node_token)
            subtoken_map.extend([map_count]*len(node_token))
            map_count += 1

    word_offset_original = word_offset
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
            #  or node.type in [":", "``", ".", ",", "XX", "X", "-LRB-", "-RRB-", "''", "HYPH"]
            continue
        node.idx = node_offset
        node_offset += 1
    constituent_sequence = [] # [(idx, start, end, type, parent idx)]
    num_internal_nodes = len(internal_nodes)
    # constituent_edge
    constituent_edge = [[0] * num_internal_nodes for _ in range(num_internal_nodes)]
    for i, node in enumerate(internal_nodes):
        # if node.type in [":", "``", ".", ",", "XX", "X", "-LRB-", "-RRB-", "''", "HYPH"]:
        #     continue
        parent_idx = node.parent.idx if node.parent else -1
        constituent_sequence.append((node.idx, node.start, node.end, node.type, parent_idx))
        if parent_idx != -1:
            constituent_edge[node.idx - node_offset_original][parent_idx - node_offset_original] = 1 #바로 아래 코드랑 보면 양방향 엣지 포함하는거임
            constituent_edge[parent_idx - node_offset_original][node.idx - node_offset_original] = 1
    # 이부분은 한계층 건너 뛰어서 엣지 ㅇ이어 주는 식임. 원래 S랑 PP는 안이어져있는데 여기서 이어줌
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
    return high_order_sequence, word_offset, node_offset, subtoken_map, subtokens


def print_parse_string(claim_list):
    for claim in claim_list:
        input_string = claim.lower()
        doc = nlp(input_string)
        sent = list(doc.sents)[0]
        print(sent)
        parse_string = sent._.parse_string
        print(parse_string)


def save(self):
    # save graphs and labels
    self.save_path = '.'
    self.mode = 'test'
    graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    save_graphs(graph_path, self.graphs, {'labels': self.labels})
    # save other information in python dict
    info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    save_info(info_path, {'num_classes': self.num_classes})

def load(self):
    # load processed data from directory `self.save_path`
    self.save_path = '.'
    self.mode = 'test'
    graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    self.graphs, label_dict = load_graphs(graph_path)
    self.labels = label_dict['labels']
    info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    self.num_classes = load_info(info_path)['num_classes']

def has_cache(self):
    # check whether there are processed data in `self.save_path`
    self.save_path = '.'
    self.mode = 'test'
    graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    return os.path.exists(graph_path) and os.path.exists(info_path)


#print_parse_string(claims)




# doc = nlp(input_string)
#
# sent = list(doc.sents)[0]
# print(sent)
# parse_string = sent._.parse_string
# print(parse_string)
#
# # 원랜 read_constituents 파트. tree.py
# constituents = []
# word_offset, node_offset = 0, 0
# constituent = []
# constituent_sequence, word_offset, node_offset, subtoken_map, subtokens = constituent_to_tree(parse_string, word_offset, node_offset)
# subtoken_map = torch.tensor(subtoken_map, dtype=torch.int64)
# print('constitutuent sequence : ', constituent_sequence) # constituent sequence 0번째가 원래 노드, 1번째가 grand parent와 grand child 관련
# print('word offset , node offset', word_offset, node_offset)
# constituent.append(constituent_sequence)
# constituents.append(constituent)
#
# # 그래프 만들기
# num_tokens = subtoken_map.size()[0] # 문장 토크나이즈 해서 나온 토큰 개수
# num_cons = sum([len(sent_cons[0]) for sent_cons in constituent]) # cons 노드 개수
# graph = dgl.graph([])
# graph.set_n_initializer(dgl.frame.zero_initializer)
# print(graph)
#
# # 그래프에 토큰 관련 추가
# graph.add_nodes(num_tokens)
# graph.ndata['unit'] = torch.zeros(num_tokens)
# graph.ndata['dtype'] = torch.zeros(num_tokens)
#
# # constituent tree 그래프
# graph.add_nodes(num_cons)
# graph.ndata['unit'][num_tokens:] = torch.ones(num_cons)
# graph.ndata['dtype'][num_tokens:] = torch.ones(num_cons)
#
#
# constituent_starts = []
# constituent_ends = []
# constituent_labels = []
# prev_root_node_id = None
# forward_edge_type, backward_edge_type = 0, 2
# constituent_start_idx = 0
# node_id_offset = 0
# num_tokens = len(subtoken_map)
# token_range = torch.arange(0, num_tokens, dtype=torch.int64)
# cons_tag2id = get_cons_tag_vocab('../../data/IAM/constituent_gold_vocab.txt')
#
#
# for high_order_sent_cons in constituent:
#     for i, sent_cons in enumerate(high_order_sent_cons):
#         for idx, start, end, label, parent_idx in sent_cons:
#             idx_nodeid = idx - constituent_start_idx + node_id_offset # 원래는 constituent_start_idx = 0, node_id_offset = 406(token id까지였음. 1063중 406이 토큰이고 이후가 node 였었음.)
#             # parent 없는 노드
#             if parent_idx == -1:
#                 if prev_root_node_id is not None:
#                     graph.add_edges(prev_root_node_id, idx_nodeid,
#                                     data={'cc_link': torch.tensor([forward_edge_type + i]),
#                                           'dtype': torch.tensor([forward_edge_type + i])})
#                     # dual GAT
#                     graph.add_edges(idx_nodeid, prev_root_node_id,
#                                     data={'cc_link': torch.tensor([backward_edge_type + i]),
#                                           'dtype': torch.tensor([backward_edge_type + i])})
#                 prev_root_node_id = idx_nodeid
#             # parent 없는 노드들
#             if parent_idx != -1:
#                 parent_idx_nodeid = parent_idx - constituent_start_idx + node_id_offset
#                 graph.add_edges(parent_idx_nodeid, idx_nodeid,
#                                 data={'cc_link': torch.tensor([forward_edge_type + i]),
#                                       'dtype': torch.tensor([forward_edge_type + i])})
#                 graph.add_edges(idx_nodeid, parent_idx_nodeid,
#                                 data={'cc_link': torch.tensor([backward_edge_type + i]),
#                                       'dtype': torch.tensor([backward_edge_type + i])})
#
#             if i == 0:
#                 # self-loop edge
#                 graph.add_edges(idx_nodeid, idx_nodeid, data={'cc_link': torch.tensor([4]),
#                                                               'dtype': torch.tensor([4])})
#                 # constituent -> token
#                 token_start = token_range[subtoken_map == start][0]
#                 token_end = token_range[subtoken_map == end][-1]
#                 graph.add_edges(idx_nodeid, token_start, data={'ct_link': torch.tensor([5]),
#                                                                'dtype': torch.tensor([5])})
#                 graph.add_edges(idx_nodeid, token_end, data={'ct_link': torch.tensor([5]),
#                                                              'dtype': torch.tensor([5])})
#                 constituent_starts.append(token_start)
#                 constituent_ends.append(token_end)
#                 constituent_labels.append(cons_tag2id[label])
#
# print(graph)
# # ndata
# # unit 0이 token 노드, 1이 cons 노드
# #print(graph.ndata)
# print('graph ndata unit',graph.ndata['unit'])
# print('graph ndata dtype', graph.ndata['dtype'])
#
# # edata
# # cc link : 4(self loop edge), node-token : 5(constituent token edge) -> 이건 grand 이런거 아니고 그냥 일반적인 parent, child 트리
# # forward edge type(cc link) : 0, backward edge type(cc link) : 2 -> 일반적인 parent child 트리
# # forward edge type(cc link) : 1, backward edge type(cc link) : 3 -> grand parent child 트리
# #print(graph.edata)
# print('graph edata cc link', graph.edata['cc_link'])
# print('graph edata ct link', graph.edata['ct_link'])
# print('graph edata dtype', graph.edata['dtype'])
#
# dgl.save_graphs('graph.dgl', graph)
# (g,), _ = dgl.load_graphs('graph.dgl')



nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('benepar', config={'model':'benepar_en3'})
input_string = 'Effects in the classroom'
input_string = input_string.lower()
doc = nlp(input_string)

sent = list(doc.sents)[0]
# print(sent)
# parse_string = sent._.parse_string
# print(parse_string)
# #
# # for tok in doc:
# #
# #     print()
# t = nltk_tree.fromstring(sent._.parse_string)
# TreeView(t)._cframe.print_to_file('output1.ps')
# os.system('convert output1.ps output1.png')
#
# t = nltk_tree.fromstring(sent._.parse_string)
# print(TreePrettyPrinter(t).text())

from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget

cf = CanvasFrame()
t = Tree.fromstring(sent._.parse_string)
tc = TreeWidget(cf.canvas(),t)
tc['node_font'] = 'arial 14 bold'
tc['leaf_font'] = 'arial 14'
tc['node_color'] = '#005990'
tc['leaf_color'] = '#3F8F57'
tc['line_color'] = '#175252'
cf.add_widget(tc,10,10) # (10,10) offsets
cf.print_to_file('tree1.ps')
cf.destroy()
os.system('convert tree1.ps tree1.png')
