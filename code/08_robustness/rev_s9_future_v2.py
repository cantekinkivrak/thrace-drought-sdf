# Run from the repository root (python code/08_robustness/rev_s9_future_v2.py [B]).
# S9 (v2): year-resampling bootstrap of the FUTURE SDF CHANGE, using exactly the
# Table-10 statistic in every replicate:
#   station cell (model, station, D): fixed AD-selected family (as in the point estimate),
#   parameters refitted on the resampled years (warm start)  ->  model median per station
#   -> change % vs the station's historical T (historical years resampled too, fixed family)
#   -> regional median across stations  = one replicate of the Table-10 statistic.
# Also records the regional-median future severity level (median across stations of the
# model-median T), the historical regional median, and a future-only-resampling variant.
import pandas as pd, numpy as np, warnings, sys, time
from scipy import stats
from multiprocessing import Pool
warnings.filterwarnings('ignore')

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 2026
DISTS = {'GEV':stats.genextreme,'Weibull3':stats.weibull_min,'LN3':stats.lognorm,
         'Gamma3':stats.gamma,'Gumbel':stats.gumbel_r,'Normal':stats.norm,'Lojistik':stats.logistic}
WINDOWS = {'2030-2059': (2030,2059), '2070-2099': (2070,2099)}
DLIST = [6, 12]; TLIST = [50, 100]
MIN_NZ = 8

fx = pd.read_csv('data/cmip6_station_series/gelecek_spi_spei12.csv')
ffit = pd.read_csv('results/future/gelecek_sdf_fits.csv')
hist = pd.read_csv('results/sdf/sdf_fits_tum.csv')
cs = pd.read_csv('data/derived/kritik_siddet_serileri_tum.csv')

def crit_series(vals, yrs, y0, y1, DMAX=12):
    crit = {}
    dry = np.isfinite(vals) & (vals < 0)
    i = 0
    while i < len(vals):
        if dry[i]:
            j = i
            while j+1 < len(vals) and dry[j+1]: j += 1
            seg = -vals[i:j+1]; L = j-i+1
            csum = np.concatenate([[0.0], np.cumsum(seg)])
            for D in range(1, min(L, DMAX)+1):
                for a in range(0, L-D+1):
                    ye = int(yrs[i+a+D-1])
                    if y0 <= ye <= y1:
                        sD = float(csum[a+D]-csum[a])
                        if sD > crit.get((ye, D), 0.0): crit[(ye, D)] = sD
            i = j+1
        else: i += 1
    return crit

def rl(dist, p, p0, T):
    pe = 1.0/T
    if pe >= (1-p0): return 0.0
    return float(np.clip(dist.ppf(1 - pe/(1-p0), *p), 0, None))

def fit_fixed(dist, y, p_full):
    nz = y[y>0]
    if len(nz) < MIN_NZ: return None, None
    p0 = (y==0).mean()
    try:
        if len(p_full) == 3: pb = dist.fit(nz, p_full[0], loc=p_full[1], scale=p_full[2])
        else: pb = dist.fit(nz, loc=p_full[-2], scale=p_full[-1])
    except Exception: return None, None
    return pb, p0

# ---- build cells ----
# future cells: (ssp, window, D, model, station) -> series, family, full-fit params
fut_cells = {}
for (model, ssp, st), g in fx.groupby(['model','ssp','name']):
    g = g.sort_values(['year','month'])
    v = g['SPEI12'].values
    if np.isnan(v).all(): continue
    for wname, (y0,y1) in WINDOWS.items():
        crit = crit_series(v, g.year.values, y0, y1)
        for D in DLIST:
            s = np.array([crit.get((y, D), 0.0) for y in range(y0, y1+1)])
            row = ffit[(ffit.model==model)&(ffit.ssp==ssp)&(ffit.name==st)&(ffit.indeks=='SPEI')&(ffit.pencere==wname)&(ffit.D==D)]
            if len(row)==0: continue   # cell not fitted in the point estimate (n<12) -> not in Table 10 either
            fam = row.dagilim.iloc[0]; dist = DISTS[fam]
            nz = s[s>0]
            assert len(nz)==row.n_nonzero.iloc[0], (model,ssp,st,wname,D,len(nz),row.n_nonzero.iloc[0])
            p_full = dist.fit(nz)
            fut_cells[(ssp,wname,D,model,st)] = (s, fam, p_full, row.T50.iloc[0], row.T100.iloc[0])
print('future cells:', len(fut_cells))

hist_cells = {}
for D in DLIST:
    for st, g in cs[(cs.k==12)&(cs.indeks=='SPEI')&(cs.D==D)].groupby('name'):
        y = g.sort_values('year').S.values
        rr = hist[(hist.k==12)&(hist.indeks=='SPEI')&(hist.D==D)&(hist.name==st)]
        if len(rr)==0: print('no historical fit:', D, st); continue
        row = rr.iloc[0]
        dist = DISTS[row.dagilim]
        hist_cells[(D,st)] = (y, row.dagilim, dist.fit(y[y>0]), row.T50, row.T100)
print('hist cells:', len(hist_cells))

# ---- point estimate check (must reproduce Table 10) ----
def table10_point():
    out = {}
    for ssp in ['ssp245','ssp585']:
        for wname in WINDOWS:
            for D in DLIST:
                for T in TLIST:
                    ch = []; lv = []
                    for st in sorted({k[4] for k in fut_cells}):
                        ts = [fut_cells[k][3 if T==50 else 4] for k in fut_cells if k[:3]==(ssp,wname,D) and k[4]==st]
                        if not ts or (D,st) not in hist_cells: continue
                        tf = np.median(ts); th = hist_cells[(D,st)][3 if T==50 else 4]
                        ch.append(100*(tf/th-1)); lv.append(tf)
                    out[(ssp,wname,D,T)] = (np.median(ch), np.median(lv), len(ch))
    return out
pt = table10_point()
for k,v in pt.items(): print('POINT', k, 'change %.1f  level %.2f  nst %d' % v)

# ---- bootstrap ----
KEYS = [(ssp,wname,D,T) for ssp in ['ssp245','ssp585'] for wname in WINDOWS for D in DLIST for T in TLIST]
STATIONS = sorted({k[4] for k in fut_cells})

def one_rep(b):
    rng = np.random.default_rng(SEED + b)
    # historical resample per (D, station)
    hb = {}
    for (D,st),(y,fam,pf,_,_) in hist_cells.items():
        yb = y[rng.integers(0, len(y), len(y))]
        pb, p0 = fit_fixed(DISTS[fam], yb, pf)
        hb[(D,st)] = None if pb is None else {T: rl(DISTS[fam], pb, p0, T) for T in TLIST}
    # future resample per cell
    fb = {}
    for key,(s,fam,pf,_,_) in fut_cells.items():
        sb = s[rng.integers(0, len(s), len(s))]
        pb, p0 = fit_fixed(DISTS[fam], sb, pf)
        fb[key] = None if pb is None else {T: rl(DISTS[fam], pb, p0, T) for T in TLIST}
    res = {}
    for (ssp,wname,D,T) in KEYS:
        ch_both = []; ch_fut = []; lv = []
        for st in STATIONS:
            ts = [fb[k][T] for k in fb if k[:3]==(ssp,wname,D) and k[4]==st and fb[k] is not None and np.isfinite(fb[k][T]) and fb[k][T] < 500]
            if not ts or (D,st) not in hist_cells: continue
            tf = np.median(ts); lv.append(tf)
            th_pt = hist_cells[(D,st)][3 if T==50 else 4]
            ch_fut.append(100*(tf/th_pt-1))
            hh = hb[(D,st)]
            if hh is not None and np.isfinite(hh[T]) and 0 < hh[T] < 500:
                ch_both.append(100*(tf/hh[T]-1))
        res[(ssp,wname,D,T)] = (np.median(ch_both) if ch_both else np.nan, np.median(ch_fut) if ch_fut else np.nan,
                                np.median(lv) if lv else np.nan, len(ch_both), len(lv))
    return res

if __name__ == '__main__':
    t0 = time.time()
    with Pool() as pool:
        reps = pool.map(one_rep, range(B), chunksize=10)
    print('bootstrap time %.0f s' % (time.time()-t0))
    rows = []
    for (ssp,wname,D,T) in KEYS:
        a = np.array([r[(ssp,wname,D,T)] for r in reps], dtype=float)
        pc, pl, nst = pt[(ssp,wname,D,T)]
        hist_med = np.median([hist_cells[(D,st)][3 if T==50 else 4] for st in STATIONS if (D,st) in hist_cells])
        ok_both = np.isfinite(a[:,0]); ok_fut = np.isfinite(a[:,1])
        rows.append({'ssp':ssp,'pencere':wname,'D':D,'T':T,
                     'tarihsel_bolge_medyan':round(hist_med,1),
                     'gelecek_bolge_medyan':round(pl,1),
                     'gelecek_5':round(np.nanpercentile(a[:,2],5),1),'gelecek_95':round(np.nanpercentile(a[:,2],95),1),
                     'degisim_%':round(pc,0),
                     'degisim_5':round(np.percentile(a[ok_both,0],5),0),'degisim_95':round(np.percentile(a[ok_both,0],95),0),
                     'degisim_5_futonly':round(np.percentile(a[ok_fut,1],5),0),'degisim_95_futonly':round(np.percentile(a[ok_fut,1],95),0),
                     'boot_medyan_degisim':round(np.nanmedian(a[:,0]),0),
                     'n_ist':nst,'kullanilabilir_rep_%':round(100*ok_both.mean(),0),
                     'ort_ist_per_rep':round(a[:,3].mean(),1)})
    df = pd.DataFrame(rows)
    df.to_csv('results/robustness/rev_future_uncertainty_v2.csv', index=False)
    print(df.to_string(index=False))
