# Tam ızgara SDF fitleri: 8 istasyon × {SPI,SPEI} × k{1,3,6,9,12} × D{1..12}
import pandas as pd, numpy as np, warnings, time
from scipy import stats
warnings.filterwarnings('ignore')

W = "/home/claude/work/"
cs = pd.read_csv(W + "kritik_siddet_serileri_tum.csv")
T_LIST = [2, 5, 10, 25, 50, 100]
CANDS = {'GEV':stats.genextreme, 'Weibull3':stats.weibull_min, 'LN3':stats.lognorm,
         'Gamma3':stats.gamma, 'Gumbel':stats.gumbel_r, 'Normal':stats.norm, 'Lojistik':stats.logistic}

def ad_stat(x, dist, params):
    x = np.sort(x); n = len(x)
    F = np.clip(dist.cdf(x, *params), 1e-9, 1-1e-9)
    i = np.arange(1, n+1)
    return float(-n - np.mean((2*i-1)*(np.log(F) + np.log(1-F[::-1]))))

def ad_crit(n): return 0.752/(1 + 0.75/n + 2.25/n**2)

def fit_best(x):
    out = []
    for nm, dist in CANDS.items():
        try:
            p = dist.fit(x)
            ad = ad_stat(x, dist, p)
            if np.isfinite(ad): out.append((nm, dist, p, ad))
        except Exception: pass
    out.sort(key=lambda t: t[3])
    return out[0] if out else None

t0 = time.time()
rows = []
for (st, fam, k), grp in cs.groupby(['name','indeks','k']):
    for D in range(1, 13):
        s = grp[grp.D==D].S.values
        nz = s[s>0]; p0 = float((s==0).mean())
        if len(nz) < 15: continue
        best = fit_best(nz)
        if best is None: continue
        nm, dist, p, ad = best
        r = {'name':st,'indeks':fam,'k':k,'D':D,'n_nonzero':len(nz),'P0':round(p0,3),
             'dagilim':nm,'AD':round(ad,3),'uygun':bool(ad < ad_crit(len(nz)))}
        for T in T_LIST:
            pexc = 1.0/T
            if pexc >= (1-p0): r[f'T{T}'] = 0.0
            else:
                F = 1 - pexc/(1-p0)
                r[f'T{T}'] = round(float(np.clip(dist.ppf(F, *p), 0, None)), 2)
        rows.append(r)
fdf = pd.DataFrame(rows)
fdf.to_csv(W + "sdf_fits_tum.csv", index=False)
print(f"Toplam fit hücresi: {len(fdf)}  ({time.time()-t0:.0f} sn)")
print(f"AD testinden geçen: {fdf.uygun.sum()} / {len(fdf)}  (%{100*fdf.uygun.mean():.0f})")
print("\nKazanan dağılım sayıları:")
print(fdf.dagilim.value_counts().to_string())
print("\nHücre sayısı (indeks × k):")
print(fdf.groupby(['indeks','k']).size().unstack(fill_value=0).to_string())

# Bölge özeti: k=12, D=12, T=100 — istasyon sıralamasına ilk bakış
sub = fdf[(fdf.k==12)&(fdf.D==12)]
piv = sub.pivot_table(index='name', columns='indeks', values='T100')
piv['SPEI_artis_%'] = (100*(piv['SPEI']-piv['SPI'])/piv['SPI']).round(1)
print("\nk=12, D=12, T=100 şiddetleri (istasyon sıralaması ilk bakış):")
print(piv.sort_values('SPEI', ascending=False).round(2).to_string())

# Sanity: aşırı uç T100 değerleri (GEV kuyruk patlaması kontrolü)
q = fdf.nlargest(5, 'T100')[['name','indeks','k','D','n_nonzero','dagilim','T100']]
print("\nEn büyük 5 T100 (kuyruk kontrolü):")
print(q.to_string(index=False))
