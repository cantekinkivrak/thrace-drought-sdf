# Bootstrap CI v2: D={3,6,9,12}, NB=200, çok çekirdekli, warm-start
import pandas as pd, numpy as np, warnings, time, os
from scipy import stats
from multiprocessing import Pool
warnings.filterwarnings('ignore')

W = "/home/claude/work/"
DISTS = {'GEV':stats.genextreme,'Weibull3':stats.weibull_min,'LN3':stats.lognorm,
         'Gamma3':stats.gamma,'Gumbel':stats.gumbel_r,'Normal':stats.norm,'Lojistik':stats.logistic}
NB = 1000
T_CI = [10, 50, 100]

cs = pd.read_csv(W + "kritik_siddet_serileri_tum.csv")
ff = pd.read_csv(W + "sdf_fits_tum.csv")
cells = ff[(ff.k==12) & (ff.D.isin([3,6,9,12]))].to_dict('records')

def rl(dist, p, p0, T):
    pe = 1.0/T
    if pe >= (1-p0): return 0.0
    return float(np.clip(dist.ppf(1 - pe/(1-p0), *p), 0, None))

def one_cell(c):
    rng = np.random.default_rng(hash((c['name'], c['indeks'], c['D'])) % 2**32)
    g = cs[(cs.name==c['name'])&(cs.indeks==c['indeks'])&(cs.k==12)&(cs.D==c['D'])]
    y = g.S.values; n = len(y)
    dist = DISTS[c['dagilim']]
    p_full = dist.fit(y[y>0])
    bs = {T: [] for T in T_CI}
    for b in range(NB):
        yb = y[rng.integers(0, n, n)]
        nz = yb[yb>0]
        if len(nz) < 10: continue
        p0b = (yb==0).mean()
        try:
            if len(p_full) == 3:
                pb = dist.fit(nz, p_full[0], loc=p_full[1], scale=p_full[2])
            else:
                pb = dist.fit(nz, loc=p_full[-2], scale=p_full[-1])
            vals = {T: rl(dist, pb, p0b, T) for T in T_CI}
            if all(np.isfinite(v) and v < 500 for v in vals.values()):
                for T in T_CI: bs[T].append(vals[T])
        except Exception:
            pass
    row = {'name':c['name'],'indeks':c['indeks'],'D':int(c['D']),'dagilim':c['dagilim'],
           'n_bs':len(bs[100]), 'T10':c['T10'], 'T50':c['T50'], 'T100':c['T100']}
    for T in T_CI:
        if len(bs[T]) >= 80:
            row[f'T{T}_alt'] = round(float(np.percentile(bs[T], 5)), 2)
            row[f'T{T}_ust'] = round(float(np.percentile(bs[T], 95)), 2)
        else:
            row[f'T{T}_alt'] = np.nan; row[f'T{T}_ust'] = np.nan
    return row

import sys
IXF = sys.argv[1] if len(sys.argv) > 1 else None
if IXF: cells = [c for c in cells if c['indeks'] == IXF]

if __name__ == '__main__':
    t0 = time.time()
    with Pool(min(8, os.cpu_count())) as pool:
        out = pool.map(one_cell, cells)
    bdf = pd.DataFrame(out)
    bdf.to_csv(W + f"bootstrap_ci_k12_{IXF or 'all'}.csv", index=False)
    print(f"Bootstrap tamam: {len(bdf)} hücre, {time.time()-t0:.0f} sn (çekirdek: {min(8, os.cpu_count())})")
    print("\nD=12 örnek:")
    print(bdf[bdf.D==12][['name','indeks','T100','T100_alt','T100_ust','n_bs']].to_string(index=False))
