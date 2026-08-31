# Paper 3 pipeline Adım 1-2: yağış dolgu makullük kontrolü + Thornthwaite PET + SPI/SPEI (1/3/6/12)
import pandas as pd, numpy as np, re
from scipy import stats
from math import gamma as GAMMA

W = "/home/claude/work/"
D3 = "/mnt/user-data/uploads/Paper 3/data/"
FLEX = ("/mnt/user-data/uploads/paper 2/files (60)/Supporting_Information_package/SI_package/data/"
        "trakya_spi_spei_flexible_1965_2024.csv")
MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
PFILES = {"Çorlu":"corlu_prcp_filled.csv","Edirne":"edirne_prcp_filled.csv","İpsala":"ipsala_prcp_filled.csv",
          "Kırklareli":"kirklareli_prcp_filled.csv","Lüleburgaz":"luleburgaz_prcp_filled.csv",
          "Sarıyer":"sariyer_prcp_filled.csv","Tekirdağ":"tekirdagp_filled.csv","Uzunköprü":"uzunkopru_prcp_filled.csv"}
NAME_MAP = {'Corlu':'Çorlu','Edirne':'Edirne','Ipsala':'İpsala','Kirklareli':'Kırklareli',
            'Luleburgaz':'Lüleburgaz','Sariyer':'Sarıyer','Tekirdag':'Tekirdağ','Uzunkopru':'Uzunköprü'}

# ---------- Veri ----------
rows = []
for st, f in PFILES.items():
    df = pd.read_csv(D3 + f)
    l = df.melt(id_vars='year', value_vars=MON, var_name='mn', value_name='precip')
    l['month'] = l['mn'].map({m:i+1 for i,m in enumerate(MON)})
    l['name'] = st
    rows.append(l[['name','year','month','precip']])
P = pd.concat(rows, ignore_index=True)
T = pd.read_csv(W + "trakya_temp_filled_v2.csv")[['name','year','month','temp']]
data = P.merge(T, on=['name','year','month']).sort_values(['name','year','month']).reset_index(drop=True)
assert len(data) == 5760 and data.isna().sum().sum() == 0

# ---------- 1) Tespit edilen 6 yağış dolgusunun makullük kontrolü (normal-oran, r² ağırlıklı) ----------
det = pd.read_csv(W + "prcp_pchip_tespit.csv")
clim_p = data.groupby(['name','month'])['precip'].mean().rename('climp')
d2 = data.merge(clim_p, on=['name','month'])
d2['ratio'] = d2.precip / d2.climp
widر = d2.pivot_table(index=['year','month'], columns='name', values='ratio')
corr_p = widر.corr()
print("Yağış oran-korelasyonları (aralık):", round(corr_p.where(~np.eye(8,dtype=bool)).min().min(),2), "-",
      round(corr_p.where(~np.eye(8,dtype=bool)).max().max(),2))
print("\nTespit edilen 6 dolgu ayının makullük kontrolü:")
for _, r in det.iterrows():
    yr, mo = map(int, r.bas.split('-')); st = r.istasyon
    others = [o for o in PFILES if o != st]
    w = corr_p.loc[st, others]**2
    rat = widر.loc[(yr, mo), others]
    est_ratio = float((rat*w).sum() / w[rat.notna()].sum())
    cl = float(clim_p.loc[(st, mo)])
    est = est_ratio * cl
    obs = float(data[(data.name==st)&(data.year==yr)&(data.month==mo)].precip.iloc[0])
    print(f"  {st:<10} {yr}-{mo:02d}: dolgu={obs:6.1f} mm | komşu-tabanlı tahmin={est:6.1f} mm | klimatoloji={cl:5.1f} mm")

# ---------- 2) Thornthwaite PET (Paper 2 notebook ile aynı formülasyon) ----------
si = pd.read_excel(D3 + "istasyon bilgileri.xlsx"); si.columns = ['name','lat_dms','lon_dms','elev']
def dms2dd(s):
    d,m,sec,h = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", s.strip()).groups()
    return (float(d)+float(m)/60+float(sec)/3600)*(-1 if h in ('S','W') else 1)
si['lat'] = si.lat_dms.map(dms2dd); si['name'] = si['name'].map(NAME_MAP)
data = data.merge(si[['name','lat']], on='name')

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
for (nm, yr), g in data.groupby(['name','year']):
    g = g.sort_values('month')
    pa.append(pd.DataFrame({'name':nm,'year':yr,'month':range(1,13),
                            'PET':thornthwaite(g.temp.values, g.lat.iloc[0])}))
data = data.merge(pd.concat(pa, ignore_index=True), on=['name','year','month'])
data['D'] = data.precip - data.PET
data[['name','year','month','temp','precip','PET','D']].to_csv(W + "trakya_TP_PET_v2.csv", index=False, float_format='%.3f')
print("\nPET tamam. Yıllık ort PET (mm):")
print(data.groupby('name').PET.sum().div(60).round(0).astype(int).to_string())

# ---------- 3) SPI (gamma) ve SPEI (log-logistic PWM, yedekli) ----------
def spi_gamma(x):
    x = np.asarray(x, float); out = np.full(len(x), np.nan); v = np.isfinite(x); xv = x[v]
    if len(xv) < 10: return out, 'yok'
    q = (xv==0).mean(); nz = xv[xv>0]
    if len(nz) < 4: return out, 'yok'
    A = np.log(nz.mean()) - np.log(nz).mean()
    sh = (1+np.sqrt(1+4*A/3))/(4*A); sc = nz.mean()/sh
    H = np.clip(q + (1-q)*stats.gamma.cdf(xv, a=sh, scale=sc), 1e-6, 1-1e-6)
    out[v] = stats.norm.ppf(H); return out, 'gamma'

def spei_ll_pwm(x):
    x = np.asarray(x, float); v = np.isfinite(x); xv = x[v]
    xs = np.sort(xv); n = len(xs); F = (np.arange(1, n+1)-0.35)/n
    w0 = xs.mean(); w1 = np.mean((1-F)*xs); w2 = np.mean((1-F)**2*xs)
    den = 6*w1 - w0 - 6*w2
    if den == 0: return None
    beta = (2*w1 - w0)/den
    try: gab = GAMMA(1+1/beta)*GAMMA(1-1/beta)
    except Exception: return None
    alpha = (w0-2*w1)*beta/gab; gamm = w0 - alpha*gab
    if not (np.isfinite(alpha) and alpha > 0 and np.isfinite(beta)): return None
    z = (xv-gamm)/alpha
    if (z <= 0).any(): return None
    return stats.norm.ppf(np.clip(1/(1+z**(-beta)), 1e-6, 1-1e-6))

def spei_robust(x):
    x = np.asarray(x, float); out = np.full(len(x), np.nan); v = np.isfinite(x); xv = x[v]
    if len(xv) < 10: return out, 'yok'
    r = spei_ll_pwm(x)
    if r is not None:
        out[v] = r; return out, 'll-pwm'
    try:  # yedek 1: fisk (log-logistic) MLE, kaydırmalı
        c, loc, sc = stats.fisk.fit(xv - xv.min() + 1.0, floc=0)
        F = np.clip(stats.fisk.cdf(xv - xv.min() + 1.0, c, loc=0, scale=sc), 1e-6, 1-1e-6)
        out[v] = stats.norm.ppf(F); return out, 'fisk-mle'
    except Exception:
        pass
    n = len(xv)  # yedek 2: ampirik (Gringorten)
    rk = stats.rankdata(xv); F = np.clip((rk-0.44)/(n+0.12), 1e-6, 1-1e-6)
    out[v] = stats.norm.ppf(F); return out, 'ampirik'

SC = [1, 3, 6, 9, 12]
out = data[['name','year','month']].copy()
fit_log = {'SPI': {}, 'SPEI': {}}
for k in SC:
    for fam, col, fn in [('SPI','precip',spi_gamma), ('SPEI','D',spei_robust)]:
        vals = np.full(len(data), np.nan)
        for nm, g in data.groupby('name'):
            g = g.sort_values(['year','month'])
            acc = g[col].rolling(k, min_periods=k).sum().values
            res = np.full(len(g), np.nan)
            for m in range(1, 13):
                mk = (g.month.values == m)
                r, method = fn(acc[mk])
                res[mk] = r
                fit_log[fam].setdefault(method, 0); fit_log[fam][method] += 1
            vals[g.index] = res
        out[f'{fam}{k}'] = vals
print("\nFit yöntem kullanımı:", fit_log)

# ---------- 4) Doğrulama ----------
print("\nDoğrulama — NaN sayıları (beklenen: 8*(k-1)):")
for k in SC:
    print(f"  k={k:>2}: SPI={out[f'SPI{k}'].isna().sum():>3} (beklenen {8*(k-1)}), SPEI={out[f'SPEI{k}'].isna().sum():>3}")
chk = out.dropna()
print("Ortalama/SS kontrolü (SPI12, SPEI12):",
      f"SPI12 μ={chk.SPI12.mean():+.3f} σ={chk.SPI12.std():.3f} | SPEI12 μ={chk.SPEI12.mean():+.3f} σ={chk.SPEI12.std():.3f}")

# Paper 2 flexible SPEI12 ile çapraz kontrol
fl = pd.read_csv(FLEX)[['name','year','month','SPI12','SPEI12']].rename(columns={'SPI12':'SPI12_p2','SPEI12':'SPEI12_p2'})
m = out.merge(fl, on=['name','year','month'])
print("\nPaper 2 (flexible) ile korelasyon — SPI12:", round(m[['SPI12','SPI12_p2']].corr().iloc[0,1], 4),
      "| SPEI12:", round(m[['SPEI12','SPEI12_p2']].corr().iloc[0,1], 4))
for st in ['İpsala','Uzunköprü','Edirne']:
    ms = m[m.name==st]
    r_all = ms[['SPEI12','SPEI12_p2']].corr().iloc[0,1]
    w76 = ms[(ms.year>=1976)&(ms.year<=1979)]
    r_76 = w76[['SPEI12','SPEI12_p2']].corr().iloc[0,1]
    dmax = (w76.SPEI12 - w76.SPEI12_p2).abs().max()
    print(f"  {st:<10} SPEI12 r(tüm)={r_all:.4f} | r(1976-79)={r_76:.4f} | maks fark(1976-79)={dmax:.2f} σ")

out.to_csv(W + "trakya_spi_spei_v2.csv", index=False, float_format='%.3f')
print("\nKaydedildi: trakya_TP_PET_v2.csv, trakya_spi_spei_v2.csv")
