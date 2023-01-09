import spacy
from spacy import displacy
from collections import Counter
import pandas as pd
import os
import csv
from pathlib import Path
from nltk.tree import Tree
from nltk.parse.corenlp import CoreNLPParser
from nltk.parse.stanford import StanfordParser
nlp = spacy.load("en_core_web_sm")
data = pd.read_csv('../../data/IAM/claims/train.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
data.columns = ['claim_label', 'topic_sentence', 'claim_sentence', 'article_id', 'stance_label']
data = data.dropna(axis=0)
data = data[data['claim_label'] == 'C']
topics = data['topic_sentence'].tolist()
claims = data['claim_sentence'].tolist()
# with open('../../data/IAM/all_claim_sentence.txt', 'r', encoding='utf-8') as txt_file:
#     all_claims = txt_file.readlines()

counter = Counter()
claim_dep = []
claim_pos = []

# doc = nlp(claims[0])
#
#
# def token_format(token):
#     return "_".join([token.orth_, token.tag_, token.dep_])
#
# def to_nltk_tree(node):
#     if node.n_lefts + node.n_rights > 0:
#         return Tree(token_format(node),
#                     [to_nltk_tree(child)
#                      for child in node.children]
#                     )
#     else:
#         return token_format(node)
# tree = [to_nltk_tree(sent.root) for sent in doc.sents]
#     # The first item in the list is the full tree
# tree[0].draw()

# # os.environ['CLASSPATH'] = '../../stanford/*'
parser = CoreNLPParser(url='http://localhost:9000')
#parser = StanfordParser(model_path="../../stanford/edu/stanford/nlp/models/lexparser/englishPCFG.caseless.ser.gz")
def nltk_spacy_tree(sent):
    doc = nlp(sent)
    def token_format(token):
        return "_".join([token.orth_, token.tag_, token.dep_])
    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(token_format(node), [to_nltk_tree(child) for child in node.children])
        else:
            return token_format(node)
    tree = [to_nltk_tree(sent.root) for sent in doc.sents]
    print(tree[0])
nltk_spacy_tree(claims[0])

def nltk_stanford_tree(sent):
    parse = parser.raw_parse(sent)
    tree = list(parse)
    print(tree[0].draw())

#nltk_stanford_tree(claims[0])

# nlp = stanfordnlp.Pipeline(processors='tokenize,pos')
# doc = nlp(claims[0])
# print(doc)

'''
디펜던스 파서 트리 그려주는 코드 
'''
# for idx, claim in enumerate(claims[:1]):
#     doc = nlp(claim)
#     sentence_spans = list(doc.sents)
#     #displacy.serve(doc, style='dep')
#
#     svg = displacy.render(sentence_spans, style='dep')
#     output_path = Path('../../data/IAM/dep_claim_img/sentence_{}.svg'.format(idx))
#     output_path.open('w', encoding='utf-8').write(svg)
    # for tok in doc:




#     sentence_dep = []
#     sentence_pos = []
#     lemma = []
#     for tok in doc:
#         sentence_dep.append(tok.dep_)
#         sentence_pos.append(tok.pos_)
#         if tok.pos_ == 'VERB':
#             lemma.append(tok.lemma_)
#     claim_dep.append(sentence_dep)
#     claim_pos.append(sentence_pos)
#     counter.update(lemma)
# print(counter)

# with open('../../data/IAM/train_claim_pos.txt', 'w', encoding='utf-8') as pos_file:
#     for pos in claim_pos:
#         pos_file.write(' '.join(pos))
#         pos_file.write('\n')
# with open('../../data/IAM/train_claim_dep.txt', 'w', encoding='utf-8') as dep_file:
#     for dep in claim_dep:
#         dep_file.write(' '.join(dep))
#         dep_file.write('\n')

# for tok in doc:
#     print(tok.text, tok.lemma_, tok.pos_, tok.tag_, tok.dep_)
#     print()