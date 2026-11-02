"""Reproducible benchmarks for a binned density estimator, Python 3.10+.
Dependencies: numpy, scipy, pandas, matplotlib. No external data required.
"""
import os, time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
OUT=os.path.dirname(os.path.abspath(__file__))
RNG=np.random.default_rng(20261031)

def pdf(x,kind):
    if kind=='trunc_exp':
        a=3.0
        return a*np.exp(-a*x)/(1-np.exp(-a))
    return 3*x*x

def sample(n,kind,rng):
    u=rng.random(n)
    if kind=='trunc_exp': return -np.log1p(-u*(1-np.exp(-3)))/3
    return np.cbrt(u)

def make_estimator(values,nbins,method):
    counts,_=np.histogram(values,bins=nbins,range=(0,1))
    h=1/nbins; centers=(np.arange(nbins)+.5)*h
    heights=counts/(len(values)*h)
    if method=='hist':
        def fn(x): return heights[np.clip((x/h).astype(int),0,nbins-1)]
    elif method=='linear':
        def fn(x): return np.interp(x,centers,heights,left=heights[0],right=heights[-1])
    else:
        # Gaussian smoothing of binned heights; fixed bandwidth 0.05 on [0,1].
        # Reflecting boundary avoids artificial lost mass. Renormalize on [0,1].
        sm=gaussian_filter1d(heights, sigma=0.05/h, mode='reflect',truncate=4)
        def fn(x): return np.interp(x,centers,sm,left=sm[0],right=sm[-1])
    return fn

rows=[]
q=2048; x=np.linspace(0,1,q)
for kind in ('trunc_exp','power'):
 for n in (2000,10000,50000):
  for nbins in (32,64,128):
   for rep in range(8):
    values=sample(n,kind,RNG)
    truth=pdf(x,kind)
    for method in ('hist','linear','gaussian_grid'):
     t0=time.perf_counter_ns()
     fn=make_estimator(values,nbins,method)
     t1=time.perf_counter_ns()
     y=fn(x)
     t2=time.perf_counter_ns()
     ise=np.trapezoid((y-truth)**2,x)
     mass=np.trapezoid(y,x)
     rows.append(dict(distribution=kind,n=n,bins=nbins,rep=rep,method=method,
        ise=ise,integral=mass,build_ms=(t1-t0)/1e6,eval_ms=(t2-t1)/1e6))
df=pd.DataFrame(rows);df.to_csv(os.path.join(OUT,'results_raw.csv'),index=False)
g=df.groupby(['distribution','n','bins','method'],as_index=False).agg(
 mean_ise=('ise','mean'),std_ise=('ise','std'),mean_build_ms=('build_ms','mean'),mean_eval_ms=('eval_ms','mean'),mean_mass=('integral','mean'))
g['mean_total_ms']=g.mean_build_ms+g.mean_eval_ms
g.to_csv(os.path.join(OUT,'results_summary.csv'),index=False)
for kind in ('trunc_exp','power'):
 s=g[(g.distribution==kind)&(g.n==10000)]
 fig,ax=plt.subplots(figsize=(6.2,3.5))
 for name in ('hist','linear','gaussian_grid'):
  d=s[s.method==name]
  ax.plot(d.bins,d.mean_ise,marker='o',label=name)
 ax.set_xlabel('Number of bins');ax.set_ylabel('Mean integrated squared error')
 ax.set_title(kind+'; n = 10000; 8 replications');ax.legend();fig.tight_layout()
 fig.savefig(os.path.join(OUT,'ise_'+kind+'.png'),dpi=180);plt.close(fig)
print(g[(g.n==10000)&(g.bins==64)].to_string(index=False))
