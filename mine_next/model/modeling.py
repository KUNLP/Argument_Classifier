from transformers import AutoTokenizer, AutoModel, AutoConfig, \
    RobertaPreTrainedModel, RobertaModel
import torch
import torch.nn as nn
from abc import ABC
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from dgl import DGLGraph


class CGATLayer(nn.Module, ABC):
    """ Constituent-Constituent GATLayer """

    def __init__(self, in_dim, feat_embed_size, out_dim, num_heads):
        super(CGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_type=None):
        z = self.fc(h)
        num_tokens, emb_size = z.size()
        z = z.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        cons_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
        cc_edge_id = g.filter_edges(lambda edges: edges.data["dtype"] == edge_type)
        self_edge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 4)
        cc_edge_id = torch.cat([cc_edge_id, self_edge_id], dim=0)
        g.nodes[cons_node_ids].data['z'] = z
        g.apply_edges(self.edge_attention, edges=cc_edge_id)
        g.pull(cons_node_ids, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')
        return h[cons_node_ids]


class CTGATLayer(nn.Module, ABC):
    """ Constituent-Token GATLayer """

    def __init__(self, in_dim, feat_embed_size, out_dim, num_heads):
        super(CTGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_type=None):
        z = self.fc(h)
        num_tokens, emb_size = z.size()
        z = z.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        token_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
        cons_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
        ct_edge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 5)
        g.nodes[cons_node_ids].data['z'] = z
        g.apply_edges(self.edge_attention, edges=ct_edge_id)
        g.pull(token_node_ids, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')
        return h[token_node_ids]


class MultiHeadGATLayer(nn.Module, ABC):
    def __init__(self, layer, in_size, out_size, feat_embed_size, num_heads, config, merge='cat', layer_norm_eps=1e-12):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        out_dim = out_size // num_heads
        self.layer = layer(in_size, feat_embed_size, out_dim, num_heads)
        self.merge = merge
        self.dropout = nn.Dropout(p=0.2)
        self.LayerNorm = nn.LayerNorm(out_size, eps=layer_norm_eps)

    def forward(self, g, o, h, edge_type=None):
        head_outs = self.layer(g, self.dropout(h), edge_type)
        num_tokens = head_outs.size()[0]
        if self.merge == 'cat':
            out = head_outs.reshape([num_tokens, -1])
        else:
            out = torch.mean(head_outs, dim=1)
        out = o + F.elu(out)
        out = self.LayerNorm(out)
        return out


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.gcn_msg = fn.copy_u(u='h', out='m')
        self.gcn_reduce = fn.sum(msg='m', out='h')
    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.updata_all(self.gcn_msg, self.gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class MultiCGNLayer(nn.Module):
    def __init__(self):
        super(MultiCGNLayer, self).__init__()
        self.hidden_size * 2 + self.cons_hidden_size
        self.layer1 = GCNLayer()


class GraphEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GraphEmbedding, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        #with g.local_scope():
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return hg


class GraphEmbedding2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GraphEmbedding2, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv3 = dglnn.GraphConv(hidden_dim, out_dim, allow_zero_in_degree=True)
        # nn.init.xavier_normal_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv2.weight)
        # nn.init.xavier_normal_(self.conv3.weight)
    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            return hg


class RobertaReflectGraphWithGrandEdgeClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.cons_hidden_size = config.cons_hidden_size
        self.roberta = RobertaModel(config)
        self.feature_size = config.feature_size
        # 그래프 둘다 쓸때
        # self.claim_layer = nn.Sequential(
        #     nn.Linear(in_features=self.hidden_size+2*self.feature_size, out_features=self.hidden_size+2*self.feature_size),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.hidden_size+2*self.feature_size, out_features=self.num_labels),
        # )
        # 그래프 하나만 할때
        self.claim_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size + self.feature_size,
                      out_features=self.hidden_size + self.feature_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size + self.feature_size, out_features=self.num_labels),
        )
        # 원래는 self.hidden_size + self.cons_hidden_size
        self.cons_type_embeddings = nn.Embedding(len(config.cons_tag2id), self.cons_hidden_size)
        # nn.init.uniform_(self.cons_type_embeddings.weight, -1.0, 1.0)
        self.softmax = nn.Softmax(dim=-1)
        self.graph_embedding = GraphEmbedding2(self.cons_hidden_size, self.cons_hidden_size, self.feature_size)

    def forward(self, idx=None, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, sim_labels=None, all_graph=None,
                constituent_labels_first=None, constituent_labels_second=None):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state

        graph_conv_first = []
        graph_conv_second = []
        for (graph_id, first, second) in zip(idx, constituent_labels_first, constituent_labels_second):
            curr_first_g = all_graph[0][int(graph_id.item())].to("cuda")
            first_mask = first != -1
            first_label = first[first_mask]
            first_cons_node_feature = self.cons_type_embeddings(first_label)
            curr_first_g_conv = self.graph_embedding(curr_first_g, first_cons_node_feature)
            graph_conv_first.append(curr_first_g_conv)

            curr_second_g = all_graph[1][int(graph_id.item())].to("cuda")
            second_mask = second != -1
            second_label = second[second_mask]
            second_cons_node_feature = self.cons_type_embeddings(second_label)
            curr_second_g_conv = self.graph_embedding(curr_second_g, second_cons_node_feature)
            graph_conv_second.append(curr_second_g_conv)

        # graph_conv_reult = torch.stack(graph_conv_reult, dim=0)
        # cls = output[:, 0, : ] -> (4, 768)
        graph_conv_first = torch.stack(graph_conv_first, dim=0)
        graph_conv_second = torch.stack(graph_conv_second, dim=0)

        cls_token = output[:, 0, :].unsqueeze(dim=1)
        #cls_graph_concat = torch.cat([cls_token, graph_conv_first, graph_conv_second], dim=-1)
        cls_graph_concat = torch.cat([cls_token, graph_conv_second], dim=-1)
        #cls_graph_concat = torch.cat([cls_token, graph_conv_second], dim=-1)

        logit = self.claim_layer(cls_graph_concat)
        logit = logit.squeeze(dim=1)
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logit, labels)
            return loss, self.softmax(logit)
        else:
            return logit


class RobertaReflectGraphClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.cons_hidden_size = config.cons_hidden_size
        self.roberta = RobertaModel(config)
        self.feature_size = config.feature_size
        self.claim_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size+2*self.feature_size, out_features=self.hidden_size+2*self.feature_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size+2*self.feature_size, out_features=self.num_labels),
        )# 원래는 self.hidden_size + self.cons_hidden_size
        self.cons_type_embeddings = nn.Embedding(len(config.cons_tag2id), self.cons_hidden_size)
        # nn.init.uniform_(self.cons_type_embeddings.weight, -1.0, 1.0)
        self.softmax = nn.Softmax(dim=-1)
        self.graph_embedding = GraphEmbedding2(self.cons_hidden_size, self.cons_hidden_size, self.feature_size)

    def forward(self, idx=None, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, sim_labels=None, all_graph=None,
                constituent_labels_first=None, constituent_labels_second=None):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state

        graph_conv_first = []
        graph_conv_second = []
        for (graph_id, first, second) in zip(idx, constituent_labels_first, constituent_labels_second):
            curr_first_g = all_graph[0][int(graph_id.item())].to("cuda")
            first_mask = first != -1
            first_label = first[first_mask]
            first_cons_node_feature = self.cons_type_embeddings(first_label)
            curr_first_g_conv = self.graph_embedding(curr_first_g, first_cons_node_feature)
            graph_conv_first.append(curr_first_g_conv)

            curr_second_g = all_graph[1][int(graph_id.item())].to("cuda")
            second_mask = second != -1
            second_label = second[second_mask]
            second_cons_node_feature = self.cons_type_embeddings(second_label)
            curr_second_g_conv = self.graph_embedding(curr_second_g, second_cons_node_feature)
            graph_conv_second.append(curr_second_g_conv)

        # graph_conv_reult = torch.stack(graph_conv_reult, dim=0)
        # cls = output[:, 0, : ] -> (4, 768)
        graph_conv_first = torch.stack(graph_conv_first, dim=0)
        graph_conv_second = torch.stack(graph_conv_second, dim=0)

        cls_token = output[:, 0, :].unsqueeze(dim=1)
        # cls_graph_concat = torch.cat([cls_token, graph_conv_first], dim=-1)
        #cls_graph_concat = torch.cat([graph_conv_first,cls_token], dim=-1)
        cls_graph_concat = torch.cat([cls_token, graph_conv_first, graph_conv_second], dim=-1)
        logit = self.claim_layer(cls_graph_concat)
        logit = logit.squeeze(dim=1)
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logit, labels)
            return loss, self.softmax(logit)
        else:
            return logit


class RobertaForClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)
        self.claim_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_labels),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, idx=None, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, sim_labels=None):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state

        logit = self.claim_layer(output[:, 0, :])
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logit, labels)
            return loss, self.softmax(logit)
        else:
            return logit


class RobertaForStanceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)
        self.claim_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_labels),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state

        logit = self.claim_layer(output[:, 0, :])
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logit, labels)
            return loss, self.softmax(logit)
        else:
            return logit


class RobertaForSTANCY(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)
        self.claim_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size+1, out_features=self.hidden_size+1),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size+1, out_features=self.num_labels),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.cosine = nn.CosineSimilarity()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, sim_labels=None):

        output_combine = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        #output_combine = output_combine.last_hidden_state
        output_combine = output_combine.pooler_output
        sent_attention_mask = (1-token_type_ids) * attention_mask
        output_sent = self.roberta(input_ids=input_ids, attention_mask=sent_attention_mask)
        #output_sent = output_sent.last_hidden_state
        output_sent = output_sent.pooler_output
        cos_sim = self.cosine(output_combine, output_sent).unsqueeze(1)
        combined = torch.cat([output_combine, cos_sim], dim=1)

        logit = self.claim_layer(combined)

        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss_bert = loss_func(logit, labels)

            loss_cosine = nn.CosineEmbeddingLoss()
            loss_claim = loss_cosine(output_combine, output_sent, sim_labels)
            loss = loss_bert + loss_claim
            return loss, self.softmax(logit)
        else:
            return logit