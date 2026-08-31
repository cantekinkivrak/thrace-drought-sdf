# S2: (a) k=12 hücrelerinde seçilen aile için parametrik bootstrap AD-GOF (B=300)
#     (b) AICc ile aile seçimi çapraz kontrolü
#     (c) TÜM 712 hücrede sabit-GEV duyarlılığı (T50/T100 farkları)
import pandas as pd, numpy as np, warnings
from scipy import stats
from multiprocessing import Pool
warnings.filterwarnings('ignore')

W = "/home/claude/work/"
cs = pd.read_csv(W + "kritik_siddet_serileri_tum.csv")
fits = pd.read_csv(W + "sdf_fits_tum.csv")

CANDS = {'GEV': stats.genextreme, 'Weibull3': stats.weibull_min, 'LN3': stats.lognorm,
         'Gamma3': stats.gamma, 'Gumbel': stats.gumbel_r, 'Normal': stats.norm,
         'Lojistik': stats.logistic}

def ad_stat(x, dist, p):
    x = np.sort(x); n = len(x)
    F = np.clip(dist.cdf(x, *p), 1e-10, 1-1e-10)
    i = np.arange(1, n+1)
    return -n - np.sum((2*i-1)*(np.log(F) + np.log(1-F[::-1])))/n

def fit_one(dist, x):
    try:
        p = dist.fit(x)
        ad = ad_stat(x, dist, p)
        ll = np.sum(dist.logpdf(x, *p))
        if not np.isfinite(ad) or not np.isfinite(ll): return None
        return p, ad, ll
    except Exception:
        return None

def cell_boot(args):
    nm, ix, D, x, B = args
    x = np.asarray(x, float)
    n = len(x)
    # AD ile aile seç (orijinal prosedür)
    best = None
    aiccs = {}
    for name, dist in CANDS.items():
        r = fit_one(dist, x)
        if r is None: continue
        p, ad, ll = r
        kpar = len(p)
        aicc = 2*kpar - 2*ll + (2*kpar*(kpar+1))/max(n-kpar-1, 1)
        aiccs[name] = aicc
        if best is None or ad < best[2]: best = (name, p, ad)
    if best is None: return None
    name, p, ad_obs = best
    dist = CANDS[name]
    aicc_best = min(aiccs, key=aiccs.get)
    # parametrik bootstrap: aynı aile yeniden kestirilerek
    rng = np.random.default_rng(42)
    cnt = 0; ok = 0
    shp = p[:-2]; l0, s0 = p[-2], p[-1]
    for b in range(B):
        xs = dist.rvs(*p, size=n, random_state=rng)
        try:
            pb = dist.fit(xs, *shp, loc=l0, scale=s0)   # ılık başlatma
            adb = ad_stat(xs, dist, pb)
            if not np.isfinite(adb): continue
        except Exception:
            continue
        ok += 1
        if adb >= ad_obs: cnt += 1
    pboot = cnt/ok if ok > 20 else np.nan
    return {'name': nm, 'indeks': ix, 'D': D, 'n': n, 'secilen': name, 'AD': round(ad_obs,3),
            'p_boot': round(pboot,3) if pboot==pboot else np.nan, 'B_ok': ok,
            'AICc_secim': aicc_best, 'uyum': name == aicc_best}

if __name__ == '__main__':
    # ---- (a)+(b): k=12 hücreleri ----
    jobs = []
    for (nm, ix, D), g in cs[cs.k==12].groupby(['name','indeks','D']):
        x = g.S.values.astype(float); x = x[x>0]
        if len(x) >= 15: jobs.append((nm, ix, D, x.tolist(), 250))
    print('k=12 hücre:', len(jobs))
    with Pool(8) as pool:
        res = [r for r in pool.map(cell_boot, jobs) if r]
    bt = pd.DataFrame(res)
    bt.to_csv(W+'rev_gof_bootstrap_k12.csv', index=False)
    v = bt.dropna(subset=['p_boot'])
    print(f'Bootstrap GOF: {len(v)} hücre | p>=0.05 geçen: {(v.p_boot>=0.05).sum()} ({(v.p_boot>=0.05).mean()*100:.0f}%)')
    print(f'p>=0.10 geçen: {(v.p_boot>=0.10).sum()} | medyan p_boot: {v.p_boot.median():.2f}')
    print(f'AICc-AD aile uyumu: {bt.uyum.mean()*100:.0f}% ({bt.uyum.sum()}/{len(bt)})')

    # ---- (c): sabit-GEV duyarlılığı, tüm hücreler ----
    T = [10, 50, 100]
    rows = []
    for (nm, ix, k, D), g in cs.groupby(['name','indeks','k','D']):
        x0 = g.S.values.astype(float)
        x = x0[x0>0]; p0 = (x0==0).mean()
        if len(x) < 15: continue
        r = fit_one(stats.genextreme, x)
        if r is None: continue
        p, ad, ll = r
        row = {'name':nm,'indeks':ix,'k':k,'D':D}
        for TT in T:
            pe = 1.0/TT
            if pe >= (1-p0): row[f'T{TT}_gev'] = 0.0
            else:
                F = 1 - pe/(1-p0)
                row[f'T{TT}_gev'] = float(np.clip(stats.genextreme.ppf(F, *p), 0, None))
        rows.append(row)
    gv = pd.DataFrame(rows)
    m = fits.merge(gv, on=['name','indeks','k','D'])
    for TT in T:
        sel = m[f'T{TT}'].values; gev = m[f'T{TT}_gev'].values
        okm = (sel>0)
        d = np.abs(gev[okm]-sel[okm])/sel[okm]*100
        print(f'T{TT}: sabit-GEV fark medyan {np.median(d):.1f}% | q90 {np.percentile(d,90):.1f}% | maks {d.max():.1f}%')
    m.to_csv(W+'rev_gev_fixed_sens.csv', index=False)
    print('OK S2')
