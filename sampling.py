"""Random-walk sampling of four-node subgraphs."""
import numpy as np
class Sampler:
    def __init__(self,adj,rng):
        self.rng=rng; a=adj.tocsr(copy=True)
        sinks=np.flatnonzero(np.diff(a.indptr)==0)
        if len(sinks):
            a=a.tolil(); a[sinks,sinks]=1; a=a.tocsr()
        self.ptr=a.indptr; self.nb=a.indices; self.degree=np.diff(a.indptr); self.n=a.shape[0]
    def step(self,nodes):
        return self.nb[self.ptr[nodes]+self.rng.integers(self.degree[nodes])]
    def unique3(self,walk):
        ordered=np.sort(walk,axis=1); keep=np.ones_like(ordered,dtype=bool)
        keep[:,1:]=ordered[:,1:]!=ordered[:,:-1]
        ordered[~keep]=self.n
        ordered.sort(axis=1)
        count=(ordered<self.n).sum(1)
        return ordered[:,:3],count
    def sample(self,seeds=None):
        seeds=np.arange(self.n) if seeds is None else np.asarray(seeds,dtype=np.int64)
        # restart=1: 12 independent one-hop traces, seeds omitted.
        walk=self.step(np.repeat(seeds[:,None],12,axis=1))
        out,count=self.unique3(walk); pending=np.flatnonzero(count<3)
        for retry in range(1,12):
            if not len(pending): break
            seed=seeds[pending]; cur=seed.copy(); active=np.ones(len(seed),dtype=bool)
            visits=[]; step=0
            # Check the max-node threshold after completing each trace.
            while active.any():
                row=np.full(len(seed),self.n,dtype=np.int64)
                cur[active]=self.step(cur[active]); row[active]=cur[active]; visits.append(row); step+=1
                restart=self.rng.random(len(seed))<.9
                if step>=20: active &= ~restart
                cur[restart]=seed[restart]
            uniq,cnt=self.unique3(np.stack(visits,axis=1))
            if retry>10:
                for i in np.flatnonzero(cnt<=2):
                    vals=uniq[i,:cnt[i]].tolist(); uniq[i]=np.array((vals*3)[:3])
            out[pending]=uniq
            pending=pending[cnt<3] if retry<=10 else np.array([],dtype=int)
        return np.column_stack([out,seeds])
