# S6: (a) gelecek T50'lere örnekleme/parametre belirsizliği (bootstrap, GEV, D=12, SPEI)
#     (b) 2030-2059'un 2024 durağan-dışı taban çizgisiyle karşılaştırılması
import pandas as pd, numpy as np, warnings
from scipy import stats
warnings.filterwarnings('ignore')
W = "/home/claude/work/"

fx = pd.read_csv(W + "gelecek_spi_spei12.csv")
WINDOWS = {'2030-2059': (2030,2059), '2070-2099': (2070,2099)}

def crit_series_D12(vals, yrs, y0, y1):
    crit = {}
    dry = np.isfinite(vals) & (vals < 0)
    i = 0; D = 12
    while i < len(vals):
        if dry[i]:
            j = i
            while j+1 < len(vals) and dry[j+1]: j += 1
            seg = -vals[i:j+1]; L = j-i+1
            csum = np.concatenate([[0.0], np.cumsum(seg)])
            if L >= D:
                for a in range(0, L-D+1):
                    ye = int(yrs[i+a+D-1])
                    if y0 <= ye <= y1:
                        sD = float(csum[a+D]-csum[a])
                        if sD > crit.get(ye, 0.0): crit[ye] = sD
            i = j+1
        else: i += 1
    return np.array([crit.get(y, 0.0) for y in range(y0, y1+1)])

def t50_gev(s):
    nz = s[s>0]; p0 = (s==0).mean()
    if len(nz) < 8: return np.nan, None
    try:
        prm = stats.genextreme.fit(nz)
    except Exception: return np.nan, None
    pe = 0.02
    if pe >= (1-p0): return 0.0, prm
    return float(np.clip(stats.genextreme.ppf(1-pe/(1-p0), *prm), 0, None)), prm

# ---- seriler ----
series = {}
for (model, ssp, st), g in fx.groupby(['model','ssp','name']):
    g = g.sort_values(['year','month'])
    v = g['SPEI12'].values
    if np.isnan(v).all(): continue
    for wname, (y0,y1) in WINDOWS.items():
        series[(model,ssp,st,wname)] = crit_series_D12(v, g.year.values, y0, y1)

B = 500
rng = np.random.default_rng(11)
out = []
for ssp in ['ssp245','ssp585']:
    for wname in WINDOWS:
        mods = sorted({k[0] for k in series if k[1]==ssp and k[3]==wname})
        med_c = {}
        boot_med = np.full((B, len(mods)), np.nan)
        for mi, model in enumerate(mods):
            sts = [k[2] for k in series if k[0]==model and k[1]==ssp and k[3]==wname]
            t50s = {}
            prms = {}
            for st in sts:
                s = series[(model,ssp,st,wname)]
                t, prm = t50_gev(s)
                t50s[st] = t; prms[st] = prm
            med_c[model] = np.nanmedian(list(t50s.values()))
            for b in range(B):
                vals = []
                for st in sts:
                    s = series[(model,ssp,st,wname)]
                    sb = s[rng.integers(0, len(s), len(s))]
                    nz = sb[sb>0]; p0 = (sb==0).mean()
                    if len(nz) < 8 or prms[st] is None: continue
                    try:
                        c0,l0,s0 = prms[st]
                        prm_b = stats.genextreme.fit(nz, c0, loc=l0, scale=s0)
                        pe = 0.02
                        t = 0.0 if pe >= (1-p0) else float(np.clip(stats.genextreme.ppf(1-pe/(1-p0), *prm_b), 0, None))
                        vals.append(t)
                    except Exception: pass
                if vals: boot_med[b, mi] = np.median(vals)
        # model-medyanının replika bazında medyanı
        rep = np.nanmedian(boot_med, axis=1)
        lo, hi = np.nanpercentile(rep, [5, 95])
        cent = np.median(list(med_c.values()))
        rng_mod = (min(med_c.values()), max(med_c.values()))
        out.append({'ssp':ssp,'pencere':wname,'T50_medyan':round(cent,1),
                    'CI5':round(lo,1),'CI95':round(hi,1),
                    'model_min':round(rng_mod[0],1),'model_maks':round(rng_mod[1],1)})
        print(f"{ssp} {wname}: T50={cent:.1f}  örnekleme CI [{lo:.1f},{hi:.1f}]  model aralığı [{rng_mod[0]:.1f},{rng_mod[1]:.1f}]")
pd.DataFrame(out).to_csv(W+'rev_future_uncertainty.csv', index=False)

# ---- (b) 2024 taban çizgisi ----
print('\n=== 2030-2059 vs 2024 NS taban ===')
ns = pd.read_csv(W + 'nsgev_sonuclari.csv')
ns12 = ns[(ns.indeks=='SPEI')&(ns.D==12)][['name','S50_1980','S50_2024']]
ch = pd.read_csv(W + 'gelecek_sdf_degisim.csv')
f12 = ch[(ch.indeks=='SPEI')&(ch.D==12)&(ch.pencere=='2030-2059')]
for ssp in ['ssp245','ssp585']:
    sub = f12[f12.ssp==ssp].merge(ns12, on='name')
    r_hist = (sub.T50_gel/sub.T50_tar - 1)*100
    r_2024 = (sub.T50_gel/sub.S50_2024 - 1)*100
    print(f'{ssp}: tarihsel-durağan tabana göre medyan {r_hist.median():+.0f}% | 2024 NS tabanına göre medyan {r_2024.median():+.0f}%  (istasyon: {len(sub)})')
print('OK S6')
