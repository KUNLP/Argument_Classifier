import dgl.nn.pytorch as dglnn
import torch.nn as nn
import dgl.data
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import torch


###
# 이건 그래프 통쨰로 분류하는 코드
###


dataset = dgl.data.GINDataset('MUTAG', False)

dataloader = GraphDataLoader(
    dataset,
    batch_size=1024,
    drop_last=False,
    shuffle=True)


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            # (batch size, 20(아마 히든사이즈))
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)


model = Classifier(7, 20, 5)
opt = torch.optim.Adam(model.parameters())
for epoch in range(20):
    for batched_graph, labels in dataloader:
        # (num nodes, 7) 아마 라벨 개수가 7개인듯
        feats = batched_graph.ndata['attr']
        logits = model(batched_graph, feats)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()