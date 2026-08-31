# Durağan SDF bootstrap CI (k=12): yıl yeniden-örnekleme, P0 + seçilmiş dağılım birlikte, %90 CI
import pandas as pd, numpy as np, warnings, time
from scipy import stats
warnings.filterwarnings('ignore')
rng = np.random.default_rng(42)

W = "/home/claude/work/"
cs = pd.read_csv(W + "kritik_siddet_serileri_tum.csv")
ff = pd.read_csv(W + "sdf_fits_tum.csv")
DISTS = {'GEV':stats.genextreme,'Weibull3':stats.weibull_min,'LN3':stats.lognorm,
         'Gamma3':stats.gamma,'Gumbel':stats.gumbel_r,'Normal':stats.norm,'Lojistik':stats.logistic}
NB = 300
T_CI = [10, 50, 100]

def rl(dist, p, p0, T):
    pe = 1.0/T
    if pe >= (1-p0): return 0.0
    return float(np.clip(dist.ppf(1 - pe/(1-p0), *p), 0, None))

t0 = time.time(); out = []
cells = ff[ff.k==12]
for _, c in cells.iterrows():
    g = cs[(cs.name==c['name'])&(cs.indeks==c['indeks'])&(cs.k==12)&(cs.D==c['D'])]
    y = g.S.values; n = len(y)
    dist = DISTS[c['dagilim']]
    bs = {T: [] for T in T_CI}
    ok = 0
    for b in range(NB):
        yb = y[rng.integers(0, n, n)]
        nz = yb[yb>0]
        if len(nz) < 10: continue
        p0b = (yb==0).mean()
        try:
            pb = dist.fit(nz)
            vals = {T: rl(dist, pb, p0b, T) for T in T_CI}
            if all(np.isfinite(v) and v < 500 for v in vals.values()):
                for T in T_CI: bs[T].append(vals[T])
                ok += 1
        except Exception:
            pass
    row = {'name':c['name'],'indeks':c['indeks'],'D':int(c['D']),'dagilim':c['dagilim'],'n_bs':ok}
    for T in T_CI:
        if len(bs[T]) >= 100:
            row[f'T{T}_alt'] = round(float(np.percentile(bs[T], 5)), 2)
            row[f'T{T}_ust'] = round(float(np.percentile(bs[T], 95)), 2)
        else:
            row[f'T{T}_alt'] = np.nan; row[f'T{T}_ust'] = np.nan
    out.append(row)

bdf = pd.DataFrame(out)
bdf = bdf.merge(cells[['name','indeks','D','T10','T50','T100']], on=['name','indeks','D'])
bdf.to_csv(W + "bootstrap_ci_k12.csv", index=False)
print(f"Bootstrap tamam: {len(bdf)} hücre, {time.time()-t0:.0f} sn")
print("\nÖrnek (D=12):")
sub = bdf[bdf.D==12][['name','indeks','T100','T100_alt','T100_ust']]
print(sub.to_string(index=False))
