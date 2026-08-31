# Adım 7: QDM sapma düzeltme + gelecek SPI/SPEI-12 (referans-dönem parametreleriyle) + gelecek SDF
import pandas as pd, numpy as np, re, warnings
from scipy import stats
from math import gamma as GAMMA
warnings.filterwarnings('ignore')

W = "/home/claude/work/"
D3 = "/mnt/user-data/uploads/Paper 3/data/"
T_LIST = [10, 50, 100]
WINDOWS = {'2030-2059': (2030, 2059), '2070-2099': (2070, 2099)}

cm = pd.read_csv(W + "cmip6_trakya_istasyon_aylik.csv")
obs = pd.read_csv(W + "trakya_TP_PET_v2.csv")

# ---------- 1) QDM (istasyon × model × takvim ayı; kalibrasyon 1965-2014) ----------
def qdm(obs_s, hist_s, fut_s, kind):
    n = len(fut_s)
    tau = stats.rankdata(fut_s) / (n + 1)
    qh = np.quantile(np.sort(hist_s), tau)
    qo = np.quantile(np.sort(obs_s), tau)
    if kind == 'tas':
        return qo + (fut_s - qh)
    ratio = np.clip(fut_s / np.maximum(qh, 0.5), 0, 5)
    return qo * ratio

rows = []
oc = obs[(obs.year >= 1965) & (obs.year <= 2014)]
for (model, st), g in cm.groupby(['model', 'name']):
    gh = g[g.exp == 'historical']
    if len(gh) == 0: continue
    for ssp in ['ssp245', 'ssp585']:
        gf = g[g.exp == ssp].sort_values(['year', 'month'])
        if len(gf) == 0: continue
        out = gf[['year', 'month']].copy()
        for var, col, kind in [('pr_mm', 'precip', 'pr'), ('tas_c', 'temp', 'tas')]:
            if gf[var].isna().all():
                out[var] = np.nan; continue
            corr = np.full(len(gf), np.nan)
            for m in range(1, 13):
                o = oc[(oc.name == st) & (oc.month == m)][col].values
                h = gh[gh.month == m][var].dropna().values
                fmask = (gf.month == m).values
                f = gf.loc[fmask, var].values
                if len(h) < 20 or np.isnan(f).all(): continue
                corr[fmask] = qdm(o, h, f, kind)
            out[var] = np.round(corr, 3)
        out['model'] = model; out['name'] = st; out['ssp'] = ssp
        rows.append(out)
fut = pd.concat(rows, ignore_index=True)
fut.to_csv(W + "cmip6_qdm_gelecek_aylik.csv", index=False)

# Sanity: ΔT ve ΔP (2070-2099 vs gözlem 1965-2014 klimatolojisi)
ob_ann = oc.groupby('name').agg(P0_=('precip', lambda x: x.sum()/50), T0_=('temp', 'mean')).reset_index()
w2 = fut[(fut.year >= 2070) & (fut.year <= 2099)]
san = w2.groupby(['model','ssp']).agg(P=('pr_mm', lambda x: x.sum()/(30*8)), T=('tas_c','mean')).reset_index()
P0r = ob_ann.P0_.mean(); T0r = ob_ann.T0_.mean()
san['dT'] = (san['T'] - T0r).round(1); san['dP_%'] = (100*(san['P']-P0r)/P0r).round(0)
print("QDM sonrası 2070-2099 değişimleri (bölge, gözleme göre):")
print(san[['model','ssp','dT','dP_%']].to_string(index=False))

# ---------- 2) Gelecek PET (Thornthwaite) ve D ----------
NAME_MAP = {'Corlu':'Çorlu','Edirne':'Edirne','Ipsala':'İpsala','Kirklareli':'Kırklareli',
            'Luleburgaz':'Lüleburgaz','Sariyer':'Sarıyer','Tekirdag':'Tekirdağ','Uzunkopru':'Uzunköprü'}
si = pd.read_excel(D3 + "istasyon bilgileri.xlsx"); si.columns = ['name','lat_dms','lon_dms','elev']
def dms2dd(s):
    d,m,sec,h = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", s.strip()).groups()
    return (float(d)+float(m)/60+float(sec)/3600)*(-1 if h in ('S','W') else 1)
si['lat'] = si.lat_dms.map(dms2dd); si['name'] = si['name'].map(NAME_MAP)
LAT = dict(zip(si.name, si.lat))

def daylight(lat, m):
    lat = np.radians(lat); J = np.array([15,45,74,105,135,162,198,228,258,288,318,344])[m-1]
    dec = 0.409*np.sin(2*np.pi/365*J - 1.39)
    return 24/np.pi*np.arccos(np.clip(-np.tan(lat)*np.tan(dec), -1, 1))
DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]
def thornthwaite(tm, lat):
    I = np.sum((np.where(tm>0, tm, 0)/5)**1.514)
    a = 6.75e-7*I**3 - 7.71e-5*I**2 + 1.792e-2*I + 0.49239
    pet = np.zeros(12)
    for i in range(12):
        if tm[i] > 0 and I > 0:
            pet[i] = 16*(10*tm[i]/I)**a * (daylight(lat, i+1)/12) * (DAYS[i]/30)
    return pet

pa = []
for (model, ssp, st, yr), g in fut.groupby(['model','ssp','name','year']):
    g = g.sort_values('month')
    if g.tas_c.isna().any() or len(g) < 12: continue
    pa.append(pd.DataFrame({'model':model,'ssp':ssp,'name':st,'year':yr,'month':range(1,13),
                            'PET':thornthwaite(g.tas_c.values, LAT[st])}))
fut = fut.merge(pd.concat(pa, ignore_index=True), on=['model','ssp','name','year','month'], how='left')
fut['Dbal'] = fut.pr_mm - fut.PET

# ---------- 3) Referans-dönem parametreleriyle gelecek SPI-12 / SPEI-12 ----------
def fit_spi_ref(x):
    q = (x==0).mean(); nz = x[x>0]
    A = np.log(nz.mean()) - np.log(nz).mean()
    sh = (1+np.sqrt(1+4*A/3))/(4*A); sc = nz.mean()/sh
    return ('gamma', q, sh, sc)

def fit_spei_ref(x):
    xs = np.sort(x); n = len(xs); F = (np.arange(1,n+1)-0.35)/n
    w0 = xs.mean(); w1 = np.mean((1-F)*xs); w2 = np.mean((1-F)**2*xs)
    den = 6*w1 - w0 - 6*w2
    if den != 0:
        beta = (2*w1-w0)/den
        try:
            gab = GAMMA(1+1/beta)*GAMMA(1-1/beta)
            alpha = (w0-2*w1)*beta/gab; gamm = w0 - alpha*gab
            if np.isfinite(alpha) and alpha > 0 and (x > gamm).all():
                return ('llpwm', alpha, beta, gamm)
        except Exception: pass
    sh = x.min() - 1.0
    c, loc, sc = stats.fisk.fit(x - sh, floc=0)
    return ('fisk', c, sc, sh)

def apply_ref(vals, prm):
    if prm[0] == 'gamma':
        _, q, sh, sc = prm
        F = np.clip(q + (1-q)*stats.gamma.cdf(vals, a=sh, scale=sc), 1e-6, 1-1e-6)
    elif prm[0] == 'llpwm':
        _, a, b, g0 = prm
        z = np.maximum((vals - g0)/a, 1e-10)
        F = np.clip(1/(1+z**(-b)), 1e-6, 1-1e-6)
    else:
        _, c, sc, sh = prm
        F = np.clip(stats.fisk.cdf(vals - sh, c, loc=0, scale=sc), 1e-6, 1-1e-6)
    return stats.norm.ppf(F)

# Gözlem referans parametreleri (1965-2024, k=12)
obs_s = obs.sort_values(['name','year','month'])
ref = {}
for st, g in obs_s.groupby('name'):
    accP = g.precip.rolling(12, min_periods=12).sum().values
    accD = (g.precip - g.PET).rolling(12, min_periods=12).sum().values
    mo = g.month.values
    for m in range(1, 13):
        sel = (mo == m) & np.isfinite(accP)
        ref[(st, m, 'SPI')] = fit_spi_ref(accP[sel])
        ref[(st, m, 'SPEI')] = fit_spei_ref(accD[sel])

idx_rows = []
for (model, ssp, st), g in fut.groupby(['model','ssp','name']):
    g = g.sort_values(['year','month']).reset_index(drop=True)
    accP = g.pr_mm.rolling(12, min_periods=12).sum().values
    accD = g.Dbal.rolling(12, min_periods=12).sum().values
    spi = np.full(len(g), np.nan); spei = np.full(len(g), np.nan)
    for m in range(1, 13):
        sel = (g.month.values == m)
        vP = accP[sel]; vD = accD[sel]
        okP = np.isfinite(vP); okD = np.isfinite(vD)
        if okP.any():
            r = np.full(len(vP), np.nan); r[okP] = apply_ref(vP[okP], ref[(st, m, 'SPI')])
            spi[sel] = r
        if okD.any():
            r = np.full(len(vD), np.nan); r[okD] = apply_ref(vD[okD], ref[(st, m, 'SPEI')])
            spei[sel] = r
    idx_rows.append(pd.DataFrame({'model':model,'ssp':ssp,'name':st,'year':g.year,'month':g.month,
                                  'SPI12':np.round(spi,3),'SPEI12':np.round(spei,3)}))
fx = pd.concat(idx_rows, ignore_index=True)
fx.to_csv(W + "gelecek_spi_spei12.csv", index=False)
print("\nGelecek indeks ortalamaları (2070-2099, bölge):")
w2x = fx[(fx.year>=2070)&(fx.year<=2099)]
print(w2x.groupby(['model','ssp'])[['SPI12','SPEI12']].mean().round(2).to_string())

# ---------- 4) Gelecek SDF (pencere bazlı, toplam olasılık) ----------
CANDS = {'GEV':stats.genextreme,'Weibull3':stats.weibull_min,'LN3':stats.lognorm,
         'Gamma3':stats.gamma,'Gumbel':stats.gumbel_r,'Normal':stats.norm,'Lojistik':stats.logistic}
def ad_stat(x, dist, p):
    x = np.sort(x); n = len(x)
    F = np.clip(dist.cdf(x, *p), 1e-9, 1-1e-9); i = np.arange(1, n+1)
    return float(-n - np.mean((2*i-1)*(np.log(F)+np.log(1-F[::-1]))))
def fit_best(x):
    out = []
    for nm, d in CANDS.items():
        try:
            p = d.fit(x); a = ad_stat(x, d, p)
            if np.isfinite(a): out.append((nm, d, p, a))
        except Exception: pass
    out.sort(key=lambda t: t[3]); return out[0] if out else None

def crit_series(vals, yrs, mos, y0, y1, DMAX=12):
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

sdf_rows = []
for (model, ssp, st), g in fx.groupby(['model','ssp','name']):
    g = g.sort_values(['year','month'])
    for ix in ['SPI12','SPEI12']:
        v = g[ix].values
        if np.isnan(v).all(): continue
        for wname, (y0, y1) in WINDOWS.items():
            crit = crit_series(v, g.year.values, g.month.values, y0, y1)
            nyr = y1-y0+1
            for D in [3, 6, 9, 12]:
                s = np.array([crit.get((y, D), 0.0) for y in range(y0, y1+1)])
                nz = s[s>0]; p0 = (s==0).mean()
                if len(nz) < 12: continue
                b = fit_best(nz)
                if b is None: continue
                nm, dist, p, ad = b
                row = {'model':model,'ssp':ssp,'name':st,'indeks':ix.replace('12',''),'pencere':wname,'D':D,
                       'n_nonzero':len(nz),'P0':round(p0,3),'dagilim':nm}
                for T in T_LIST:
                    pe = 1.0/T
                    row[f'T{T}'] = 0.0 if pe >= (1-p0) else round(float(np.clip(dist.ppf(1-pe/(1-p0), *p), 0, None)), 2)
                sdf_rows.append(row)
gs = pd.DataFrame(sdf_rows)
gs.to_csv(W + "gelecek_sdf_fits.csv", index=False)
print(f"\nGelecek SDF hücreleri: {len(gs)}")

# ---------- 5) Tarihsel ile karşılaştırma (değişim tablosu) ----------
hist = pd.read_csv(W + "sdf_fits_tum.csv")
hist = hist[(hist.k==12) & (hist.D.isin([3,6,9,12]))][['name','indeks','D','T10','T50','T100']]
agg = gs.groupby(['ssp','pencere','indeks','name','D'])[['T10','T50','T100']].median().reset_index()  # model medyanı
cmp_ = agg.merge(hist, on=['name','indeks','D'], suffixes=('_gel','_tar'))
for T in T_LIST:
    cmp_[f'dT{T}_%'] = (100*(cmp_[f'T{T}_gel']-cmp_[f'T{T}_tar'])/cmp_[f'T{T}_tar'].replace(0, np.nan)).round(0)
cmp_.to_csv(W + "gelecek_sdf_degisim.csv", index=False)
print("\nBölge medyanı değişim (SPEI, T=50, model-medyan, % tarihsele göre):")
piv = cmp_[cmp_.indeks=='SPEI'].groupby(['ssp','pencere','D'])['dT50_%'].median().unstack()
print(piv.to_string())
