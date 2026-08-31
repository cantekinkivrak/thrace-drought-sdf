# S4-doğrulama + S5 (düzeltilmiş): orijinal pipeline standardizasyon fonksiyonlarıyla
import pandas as pd, numpy as np, re, warnings
from scipy import stats
from math import gamma as GAMMA
warnings.filterwarnings('ignore')
W = "/home/claude/work/"

tp = pd.read_csv(W + "trakya_TP_PET_v2.csv")
si = pd.read_excel('/mnt/user-data/uploads/Paper 3/data/istasyon bilgileri.xlsx')
si.columns = ['name','lat_dms','lon_dms','elev']
def dms2dd(s):
    d,m,sec,h = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", s.strip()).groups()
    return float(d)+float(m)/60+float(sec)/3600
si['lat'] = si.lat_dms.map(dms2dd)
NM = {'Corlu':'Çorlu','Edirne':'Edirne','Ipsala':'İpsala','Kirklareli':'Kırklareli',
      'Luleburgaz':'Lüleburgaz','Sariyer':'Sarıyer','Tekirdag':'Tekirdağ','Uzunkopru':'Uzunköprü'}
si['name'] = si['name'].map(NM)
LAT = dict(zip(si.name, si.lat))

DIM = {1:31,2:28.25,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
MID = {1:15,2:46,3:75,4:105,5:136,6:166,7:197,8:228,9:258,10:289,11:319,12:350}
def ra_mj(lat_deg, m):
    phi=np.radians(lat_deg); J=MID[m]
    dr=1+0.033*np.cos(2*np.pi*J/365); dec=0.409*np.sin(2*np.pi*J/365-1.39)
    ws=np.arccos(np.clip(-np.tan(phi)*np.tan(dec),-1,1))
    return (24*60/np.pi)*0.0820*dr*(ws*np.sin(phi)*np.sin(dec)+np.cos(phi)*np.cos(dec)*np.sin(ws))
tp['PET_ou'] = [max(0.408*ra_mj(LAT[n],m)*(t+5)/100.0,0)*DIM[m] if t+5>0 else 0.0
                for n,t,m in zip(tp.name,tp.temp,tp.month)]
tp['B_th'] = tp.precip - tp.PET
tp['B_ou'] = tp.precip - tp.PET_ou

# ---- orijinal fonksiyonlar (pipeline_step12 ile birebir) ----
def spei_ll_pwm_params(xv):
    xs = np.sort(xv); n = len(xs); F = (np.arange(1,n+1)-0.35)/n
    w0=xs.mean(); w1=np.mean((1-F)*xs); w2=np.mean((1-F)**2*xs)
    den = 6*w1 - w0 - 6*w2
    if den == 0: return None
    beta = (2*w1-w0)/den
    try: gab = GAMMA(1+1/beta)*GAMMA(1-1/beta)
    except Exception: return None
    alpha = (w0-2*w1)*beta/gab; gamm = w0 - alpha*gab
    if not (np.isfinite(alpha) and alpha>0 and np.isfinite(beta)): return None
    return alpha, beta, gamm

def spei_apply(params, x):
    alpha, beta, gamm = params
    z = (x-gamm)/alpha
    F = np.where(z>0, 1/(1+np.where(z>0,z,1.0)**(-beta)), 1e-6)
    return stats.norm.ppf(np.clip(F, 1e-6, 1-1e-6))

def spei_fit_apply(x_cal, x_all):
    """kalibrasyonda parametre, tüm seriye uygulama; ll-pwm -> fisk yedeği"""
    xc = x_cal[np.isfinite(x_cal)]
    if len(xc) < 15: return None, None
    p = spei_ll_pwm_params(xc)
    if p is not None:
        z = spei_apply(p, x_all)
        if np.nanstd(z[np.isfinite(x_all)]) > 0.5: return z, 'll-pwm'
    try:
        sh = xc.min() - 1.0
        c, loc, sc = stats.fisk.fit(xc - sh, floc=0)
        F = np.clip(stats.fisk.cdf(np.maximum(x_all - sh, 1e-9), c, loc=0, scale=sc), 1e-6, 1-1e-6)
        return stats.norm.ppf(F), 'fisk'
    except Exception:
        return None, None

def spi_fit_apply(x_cal, x_all):
    xc = x_cal[np.isfinite(x_cal)]
    q = (xc<=0).mean(); nz = xc[xc>0]
    if len(nz) < 10: return None
    A = np.log(nz.mean()) - np.log(nz).mean()
    sh = (1+np.sqrt(1+4*A/3))/(4*A); sc = nz.mean()/sh
    H = np.where(x_all<=0, q, q + (1-q)*stats.gamma.cdf(np.maximum(x_all,0), a=sh, scale=sc))
    return stats.norm.ppf(np.clip(H, 1e-6, 1-1e-6))

def build(col, calmax=None, k=12):
    """calmax=None -> tam kayıt kalibrasyonu; aksi hâlde yıl<=calmax"""
    rows = []; engines = []
    for nm, g in tp.groupby('name'):
        g = g.sort_values(['year','month']).reset_index(drop=True)
        a = g[col].rolling(k).sum().values
        yrs = g.year.values
        z = np.full(len(g), np.nan)
        for m in range(1,13):
            idx = (g.month==m).values & np.isfinite(a)
            if idx.sum() < 20: continue
            xa = a[idx]; ya = yrs[idx]
            xc = xa if calmax is None else xa[ya<=calmax]
            zz, eng = spei_fit_apply(xc, xa)
            if zz is None: continue
            engines.append(eng)
            z[idx] = np.clip(zz, -4.75, 4.75)
        gg = g[['name','year','month']].copy(); gg['val']=z; rows.append(gg)
    return pd.concat(rows), engines

def build_spi(calmax=None, k=12):
    rows = []
    for nm, g in tp.groupby('name'):
        g = g.sort_values(['year','month']).reset_index(drop=True)
        a = g['precip'].rolling(k).sum().values
        yrs = g.year.values
        z = np.full(len(g), np.nan)
        for m in range(1,13):
            idx = (g.month==m).values & np.isfinite(a)
            if idx.sum() < 20: continue
            xa = a[idx]; ya = yrs[idx]
            xc = xa if calmax is None else xa[ya<=calmax]
            zz = spi_fit_apply(xc, xa)
            if zz is None: continue
            z[idx] = np.clip(zz, -4.75, 4.75)
        gg = g[['name','year','month']].copy(); gg['val']=z; rows.append(gg)
    return pd.concat(rows)

sp = pd.read_csv(W + "trakya_spi_spei_v2.csv")
base = sp[['name','year','month','SPI12','SPEI12']]

# ===== S4 (düzeltilmiş motorla) =====
ou, eng = build('B_ou', None)
from collections import Counter
print('Oudin SPEI motor dağılımı:', Counter(eng))
m4 = base.merge(ou.rename(columns={'val':'SPEI12_ou'}), on=['name','year','month']).dropna()
cors = m4.groupby('name').apply(lambda g: np.corrcoef(g.SPEI12,g.SPEI12_ou)[0,1])
print('Th-Ou korelasyon: medyan %.3f (%.3f-%.3f)' % (cors.median(),cors.min(),cors.max()))
m4['dekad']=(m4.year//10)*10
dth = m4.groupby('dekad').apply(lambda g: (g.SPEI12-g.SPI12).mean())
dou = m4.groupby('dekad').apply(lambda g: (g.SPEI12_ou-g.SPI12).mean())
print('Dekadal SPEI−SPI (Th | Ou):')
for dd in dth.index: print(f'  {dd}s: {dth[dd]:+.2f} | {dou[dd]:+.2f}')
for esik in [-1.0,-1.5,-2.0]:
    a=(m4.SPI12<=esik).sum(); b=(m4.SPEI12<=esik).sum(); c=(m4.SPEI12_ou<=esik).sum()
    print(f'  ≤{esik}: SPI={a} SPEI-Th={b} ({(b-a)/max(a,1)*100:+.0f}%) SPEI-Ou={c} ({(c-a)/max(a,1)*100:+.0f}%)')
m4.to_csv(W+'rev_spei_oudin12.csv', index=False)

# ===== S5: erken kalibrasyon (<=1994) =====
ec_spei, eng2 = build('B_th', 1994)
ec_spi = build_spi(1994)
print('\nErken-kal SPEI motoru:', Counter(eng2))
m5 = ec_spi.rename(columns={'val':'SPI12_ec'}).merge(
     ec_spei.rename(columns={'val':'SPEI12_ec'}), on=['name','year','month']).dropna()
m5['dekad']=(m5.year//10)*10
dec_ = m5.groupby('dekad').apply(lambda g: (g.SPEI12_ec-g.SPI12_ec).mean())
print('Erken-kalibrasyon dekadal SPEI−SPI:')
for dd in dec_.index: print(f'  {dd}s: {dec_[dd]:+.2f}')
m5.to_csv(W+'rev_earlycal12.csv', index=False)
print('OK S4+S5')
