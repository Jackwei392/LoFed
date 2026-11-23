"""Train LoFed and score the nodes in each client."""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
    assert not torch.cuda.is_available()

from model import Model, aggregate
from sampling import Sampler
from utils import ROOT, fuse, neighbor_dissimilarity, read_graph, read_partition, wmetric

DATASETS = tuple(
    f'{name}_{protocol}'
    for name in ('pubmed', 'acm', 'dblp', 'cs', 'corafull')
    for protocol in ('inj', 'vgod')
)


def batches(order):
    n = len(order)
    if n == 1:
        raise ValueError('singleton client cannot form a negative pair')
    for offset in range(0, n, 300):
        if n - offset == 1:
            continue
        end = offset + 300
        if n - end == 1:
            end = n
        yield order[offset:end]


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


def prepare(adj, x, parts, device, rng):
    clients = []
    for ids in parts:
        a = adj[ids][:, ids].tocsr()
        rowsum = np.asarray(a.sum(1)).ravel()
        inverse = np.zeros_like(rowsum, dtype=np.float32)
        np.power(rowsum, -.5, out=inverse, where=rowsum != 0)
        normalized = sp.diags(inverse) @ a @ sp.diags(inverse) + sp.eye(len(ids), dtype=np.float32)
        feat = x[ids].copy()
        sums = feat.sum(axis=1)
        inverse_x = np.zeros_like(sums)
        np.divide(1., sums, out=inverse_x, where=sums != 0)
        feat *= inverse_x[:, None]
        clients.append({
            'ids': ids,
            'x': torch.from_numpy(feat).to(device),
            'a': torch.from_numpy(np.asarray(normalized.toarray(), dtype=np.float32)).to(device),
            'sampler': Sampler(a, rng),
            'raw_a': a,
        })
    return clients


def batch(client, sub, idx):
    device = client['x'].device
    nodes = torch.as_tensor(sub[idx], dtype=torch.long, device=device)
    b = len(idx)
    ba = client['a'][nodes[:, :, None], nodes[:, None, :]]
    bf = client['x'][nodes]
    zero_row = torch.zeros((b, 1, 4), device=device)
    zero_col = torch.zeros((b, 5, 1), device=device)
    zero_col[:, -1] = 1
    ba = torch.cat((torch.cat((ba, zero_row), dim=1), zero_col), dim=2)
    bf = torch.cat((bf[:, :-1], torch.zeros((b, 1, bf.shape[-1]), device=device), bf[:, -1:]), dim=1)
    return bf, ba


def train(global_model, clients, args):
    for rnd in range(1, args.rounds + 1):
        states = []
        for client in clients:
            model = copy.deepcopy(global_model)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
            n = len(client['ids'])
            for epoch in range(args.epochs):
                order = list(range(n))
                random.shuffle(order)
                sub = client['sampler'].sample()
                for idx in batches(order):
                    x, a = batch(client, sub, idx)
                    opt.zero_grad()
                    loss = model.loss(x, a)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('DIVERGED: nonfinite loss')
                    loss.backward()
                    opt.step()
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            del model, opt
        state = aggregate(states, [len(c['ids']) for c in clients])
        global_model.load_state_dict(state)
        print(f'round {rnd}/{args.rounds} complete', flush=True)


def score(model, clients, samples):
    collected, ids, cids = [], [], []
    with torch.no_grad():
        for cid, client in enumerate(clients):
            n = len(client['ids'])
            sums = torch.zeros((1, n), dtype=torch.float64, device=client['x'].device)
            for sample in range(samples):
                order = list(range(n))
                random.shuffle(order)
                sub = client['sampler'].sample()
                for idx in batches(order):
                    x, a = batch(client, sub, idx)
                    values = model.score(x, a)
                    if not torch.isfinite(values).all():
                        raise FloatingPointError('DIVERGED: nonfinite score')
                    sums[0, idx] += values.double()
            collected.append((sums / samples).float().cpu().numpy())
            ids.append(client['ids'])
            cids.append(np.full(n, cid, dtype=np.int64))
            print(f'scored client {cid + 1}/{len(clients)}', flush=True)
    return np.concatenate(ids), np.concatenate(cids), np.concatenate(collected, axis=1)[0]


def evaluate(dataset, nodes, clients, scores):
    with np.load(ROOT / 'dataset' / f'{dataset}.npz', allow_pickle=False) as data:
        y = data['anomaly'].astype(np.int64)
        mask = data['eval_mask'].astype(bool)
    auroc, ap = wmetric(scores, clients, y[nodes], mask[nodes])
    return {'W_AUROC': auroc, 'W_AP': ap}


def write_with_retry(operation):
    for attempt in range(21):
        try:
            return operation()
        except PermissionError:
            if attempt == 20:
                raise
            time.sleep(.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--clients', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()

    partition = ROOT / 'partitions' / f'{args.dataset}_metis_rho00_K{args.clients}_seed0.npz'
    if not partition.is_file():
        parser.error(f'partition unavailable: {partition.name}')
    if min(args.rounds, args.epochs, args.samples, args.hidden) < 1:
        parser.error('rounds, epochs, samples and hidden must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA_UNAVAILABLE')
    device = torch.device(args.device)
    out = args.out if args.out is not None else ROOT / 'results' / f'{args.dataset}_K{args.clients}_seed{args.seed}'

    rng = seed_all(args.seed)
    adj, features = read_graph(args.dataset)
    parts = read_partition(args.dataset, args.clients)
    clients = prepare(adj, features, parts, device, rng)
    model = Model(clients[0]['x'].shape[1], args.hidden).to(device)
    train(model, clients, args)

    rng = seed_all(args.seed)
    for client in clients:
        client['sampler'].rng = rng
    model.eval()
    nodes, client_ids, s_fed = score(model, clients, args.samples)
    nbr = np.empty(adj.shape[0])
    nbr[nodes] = np.concatenate([
        neighbor_dissimilarity(adj[ids][:, ids].tocsr(), features[ids])
        for ids in parts
    ])
    s_local = nbr[nodes]
    s_fused = fuse(client_ids, s_fed, s_local)
    if not np.isfinite(s_local).all() or not np.isfinite(s_fused).all():
        raise FloatingPointError('DIVERGED: nonfinite score')
    metrics = evaluate(args.dataset, nodes, client_ids, s_fused)

    write_with_retry(lambda: out.mkdir(parents=True, exist_ok=True))
    write_with_retry(lambda: np.savez_compressed(
        out / 'scores.npz', global_node=nodes, client=client_ids,
        s_local=s_local, s_fed=s_fed, s_fused=s_fused,
    ))
    write_with_retry(lambda: (out / 'metrics.json').write_text(
        json.dumps(metrics, indent=2) + '\n', encoding='utf-8',
    ))
    if args.save_model:
        write_with_retry(lambda: torch.save(model.state_dict(), out / 'model.pt'))
    print(f"W-AUROC: {metrics['W_AUROC']:.12f}")
    print(f"W-AP: {metrics['W_AP']:.12f}")


if __name__ == '__main__':
    main()
