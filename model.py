"""Contrastive graph encoder and weighted federated aggregation."""
import torch
from torch import nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.fc = nn.Linear(inputs, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.act = nn.PReLU()
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, a):
        return self.act(torch.bmm(a, self.fc(x)) + self.bias)


class Disc(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden, hidden, 1)
        nn.init.xavier_uniform_(self.bilinear.weight)
        nn.init.zeros_(self.bilinear.bias)

    def forward(self, c, h):
        if len(c) < 2:
            raise ValueError('discriminator cannot form a singleton negative')
        neg = torch.cat((c[-2:-1], c[:-1]), 0)
        return self.bilinear(h, c), self.bilinear(h, neg)


class Model(nn.Module):
    def __init__(self, inputs, hidden=64):
        super().__init__()
        self.gcn = GCN(inputs, hidden)
        self.disc = Disc(hidden)

    def embeddings(self, x, a):
        v = self.gcn(x, a)
        c = v[:, :-1].mean(1)
        h = v[:, -1]
        return c, h

    @staticmethod
    def bce(pos, neg):
        return F.binary_cross_entropy_with_logits(
            torch.cat((pos, neg)), torch.cat((torch.ones_like(pos), torch.zeros_like(neg)))
        )

    def loss(self, x, a):
        emb = self.embeddings(x, a)
        return self.bce(*self.disc(emb[0], emb[1]))

    def score(self, x, a):
        e = self.embeddings(x, a)
        return 1 - F.cosine_similarity(e[1], e[0], dim=1, eps=1e-6)


def aggregate(states, weights):
    total = sum(weights)
    return {k: sum((state[k] * float(w / total) for state, w in zip(states, weights))) for k in states[0]}
