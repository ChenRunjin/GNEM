import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MeanAggregator(nn.Module):

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GateGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GateGraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
            torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.update_gate = nn.Linear(in_dim+out_dim, out_dim)
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        u = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        u = u + self.bias
        update_weight = torch.sigmoid(self.update_gate(torch.cat([u,features],dim=2)))
        out = update_weight * torch.tanh(u) + (1 - update_weight)*features
        out = F.relu(out)
        return out


class gcn(nn.Module):
    def __init__(self, dims, dropout=0.0):
        super(gcn, self).__init__()
        self.convs = []
        self.layers = len(dims)-1
        self.dropout = nn.Dropout(dropout)

        for i in range(len(dims)-1):
            self.convs.append(GateGraphConv(dims[i], dims[i + 1], MeanAggregator))
        self.convs = nn.ModuleList(self.convs)

        self.classifier = nn.Sequential(
            nn.Linear(dims[-1], dims[-1]),
            nn.PReLU(dims[-1]),
            nn.Linear(dims[-1], 2))

    def forward(self, x, A):
        x = self.dropout(x)

        for c in self.convs:
            x = c(x, A)

        dout = x.size(-1)
        x=x.view(-1,dout)
        x = self.dropout(x)
        pred = self.classifier(x)

        return pred











