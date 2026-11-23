"""Graph readers and client-level scores."""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm, rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent


def read_graph(name):
    with np.load(ROOT / 'dataset' / f'{name}.npz', allow_pickle=False) as z:
        shape = tuple(z['adj_shape'])
        adj = sp.csr_matrix((z['adj_data'], z['adj_indices'], z['adj_indptr']), shape=shape)
        features = z['features']
    return adj, features


def read_partition(name, clients):
    path = ROOT / 'partitions' / f'{name}_metis_rho00_K{clients}_seed0.npz'
    with np.load(path, allow_pickle=False) as z:
        return [z[f'client_{i}'] for i in range(clients)]


def neighbor_dissimilarity(adjacency, raw_features):
    adjacency = adjacency.tocsr()
    raw = np.asarray(raw_features, dtype=np.float32)
    row = raw.copy()
    sums = row.sum(axis=1)
    inverse = np.zeros_like(sums)
    np.divide(1., sums, out=inverse, where=sums != 0)
    row *= inverse[:, None]
    x = row.astype(np.float64)
    degrees = np.diff(adjacency.indptr).astype(np.float64)
    neighbors = adjacency.dot(x)
    np.divide(neighbors, degrees[:, None], out=neighbors, where=degrees[:, None] != 0)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(neighbors, axis=1)
    nbr = 1.0 - np.einsum('ij,ij->i', x, neighbors) / np.maximum(denominator, 1e-12)
    nbr[degrees == 0] = 0.
    return nbr


def rank_normal(v):
    return norm.ppf((rankdata(v, method="average") - 0.5) / len(v))


def fuse(client, a, b):
    out = np.zeros(len(client))
    for c in np.unique(client):
        s = client == c
        out[s] = rank_normal(np.asarray(a[s], dtype=np.float64)) + rank_normal(np.asarray(b[s], dtype=np.float64))
    return out


def wmetric(score, client, y, mask, typ=None):
    vals, ws = [], []
    for c in np.unique(client):
        use = (client == c) & mask
        if y[use].min() == y[use].max():
            continue
        vals.append((roc_auc_score(y[use], score[use]), average_precision_score(y[use], score[use]))); ws.append(int(use.sum()))
    out = [float(np.average([v[0] for v in vals], weights=ws)), float(np.average([v[1] for v in vals], weights=ws))]
    if typ is not None:
        for code in (1, 2):
            v, w = [], []
            for c in np.unique(client):
                use = (client == c) & mask & ((y == 0) | (typ == code))
                if y[use].min() == y[use].max():
                    continue
                v.append(roc_auc_score(y[use], score[use])); w.append(int(use.sum()))
            out.append(float(np.average(v, weights=w)))
    return tuple(out)
