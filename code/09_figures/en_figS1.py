# Adım 4: Pilot SDF (Kırklareli) — 7 aday dağılım + AD seçimi + toplam olasılık dönüş seviyeleri
# + Homojenlik testleri (Pettitt, SNHT) tüm istasyonlar
import pandas as pd, numpy as np, warnings
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

W = "/home/claude/work/"
cs = pd.read_csv(W + "kritik_siddet_serileri.csv")
T_LIST = [2, 5, 10, 25, 50, 100]
CANDS = {
    'GEV':      stats.genextreme,
    'Weibull3': stats.weibull_min,
    'LN3':      stats.lognorm,
    'Gamma3':   stats.gamma,
    'Gumbel':   stats.gumbel_r,
    'Normal':   stats.norm,
    'Lojistik': stats.logistic,
}

def ad_stat(x, dist, params):
    x = np.sort(x); n = len(x)
    F = np.clip(dist.cdf(x, *params), 1e-9, 1-1e-9)
    i = np.arange(1, n+1)
    return float(-n - np.mean((2*i-1) * (np.log(F) + np.log(1 - F[::-1]))) )

def ad_crit(n):     # tezdeki Eş. 5.41 (α=0.05)
    return 0.752 / (1 + 0.75/n + 2.25/n**2)

def fit_best(x):
    out = []
    for nm, dist in CANDS.items():
        try:
            if nm in ('Gumbel','Normal','Lojistik'):
                p = dist.fit(x)
            else:
                p = dist.fit(x)          # 3 parametreli serbest fit
            ad = ad_stat(x, dist, p)
            if np.isfinite(ad): out.append((nm, dist, p, ad))
        except Exception:
            pass
    out.sort(key=lambda t: t[3])
    return out[0] if out else None

def return_levels(x, p0, dist, params):
    lv = {}
    for T in T_LIST:
        pexc = 1.0/T
        if pexc >= (1 - p0):
            lv[T] = 0.0
        else:
            F = 1 - pexc/(1 - p0)
            lv[T] = float(np.clip(dist.ppf(F, *params), 0, None))
    return lv

# ---------- Pilot: Kırklareli ----------
fits = []
curves = {}
for ix in ['SPI12','SPEI12']:
    for D in range(1, 13):
        s = cs[(cs.name=='Kırklareli')&(cs['index']==ix)&(cs.D==D)].S.values
        nz = s[s > 0]; n = len(s); p0 = (s==0).mean()
        if len(nz) < 15: continue
        best = fit_best(nz)
        if best is None: continue
        nm, dist, p, ad = best
        lv = return_levels(nz, p0, dist, p)
        fits.append({'index':ix,'D':D,'n_nonzero':len(nz),'P0':round(p0,3),
                     'dagilim':nm,'AD':round(ad,3),'AD_krit':round(ad_crit(len(nz)),3),
                     'uygun':ad < ad_crit(len(nz)),
                     **{f'T{T}':round(lv[T],2) for T in T_LIST}})
        curves[(ix, D)] = lv
fdf = pd.DataFrame(fits)
fdf.to_csv(W + "kirklareli_sdf_fits.csv", index=False)
print("KIRKLARELİ PİLOT FİT ÖZETİ:")
print(fdf.to_string(index=False))

# ---------- Figür: SDF eğrileri ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
cmap = plt.get_cmap('Reds')
tcolors = {T: cmap(0.30 + 0.65*i/(len(T_LIST)-1)) for i, T in enumerate(T_LIST)}
for ax, ix in zip(axes, ['SPI12','SPEI12']):
    Ds = sorted({d for (i2, d) in curves if i2==ix})
    for T in T_LIST:
        y = [curves[(ix, D)][T] for D in Ds]
        ax.plot(Ds, y, marker='o', ms=3.5, lw=1.8, color=tcolors[T], label=f'T = {T} yr')
    ax.set_title(f'{"A" if ix=="SPI12" else "B"} — {ix}-based SDF (Kırklareli)', fontsize=11, loc='left')
    ax.set_xlabel('Drought duration D (months)')
    ax.set_xticks(Ds)
    ax.grid(axis='y', color='#E5E7EB', lw=0.6); ax.set_axisbelow(True)
    for sp_ in ['top','right']: ax.spines[sp_].set_visible(False)
axes[0].set_ylabel('Critical drought severity S')
axes[1].legend(fontsize=8.5, frameon=False, loc='upper left')
plt.tight_layout()
plt.savefig(W + "figs_EN/FigS1_kirklareli_sdf.png", dpi=300, bbox_inches='tight')
print("\nFigür kaydedildi: kirklareli_sdf.png")

# ---------- Homojenlik testleri ----------
tp = pd.read_csv(W + "trakya_TP_PET_v2.csv")
ann = tp.groupby(['name','year']).agg(T=('temp','mean'), P=('precip','sum')).reset_index()

def pettitt(x):
    n = len(x); r = stats.rankdata(x)
    U = [2*np.sum(r[:t+1]) - (t+1)*(n+1) for t in range(n)]
    K = np.max(np.abs(U)); t0 = int(np.argmax(np.abs(U)))
    p = 2*np.exp(-6*K**2/(n**3 + n**2))
    return t0, min(p, 1.0)

def snht(x):
    n = len(x); z = (x - x.mean())/x.std(ddof=1)
    Tk = [k*np.mean(z[:k])**2 + (n-k)*np.mean(z[k:])**2 for k in range(1, n)]
    return int(np.argmax(Tk))+1, float(np.max(Tk))

SNHT_CRIT = 8.1   # n≈60, α=0.05 (yaklaşık; Khaliq & Ouarda 2007)
rows = []
for st, g in ann.groupby('name'):
    g = g.sort_values('year')
    for var, x in [('Sıcaklık', g['T'].values.astype(float)), ('Yağış', g['P'].values.astype(float))]:
        t0, p = pettitt(x); k, Tmax = snht(x)
        rows.append({'istasyon':st,'değişken':var,
                     'Pettitt_yıl':int(g.year.iloc[t0]),'Pettitt_p':round(p,4),
                     'SNHT_yıl':int(g.year.iloc[k-1]),'SNHT_T':round(Tmax,1),
                     'SNHT_anlamlı':Tmax > SNHT_CRIT})
hom = pd.DataFrame(rows)
hom.to_csv(W + "homojenlik.csv", index=False)
print("\nHOMOJENLİK TESTLERİ (yıllık ortalamalar):")
print(hom.to_string(index=False))
