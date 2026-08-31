# S4: PET yöntem duyarlılığı — Thornthwaite vs Oudin (yalnız Tmean gerektirir)
# Çıktılar: SPEI12 korelasyonu, dekadal SPEI-SPI ıraksaması (iki PET), ekstrem ay sayıları,
#           D=12 kritik şiddet serilerinde MK karşılaştırması + T50/T100 (k=12, D=12)
import pandas as pd, numpy as np, re, warnings
from scipy import stats
warnings.filterwarnings('ignore')
W = "/home/claude/work/"

# ---- veri ----
tp = pd.read_csv(W + "trakya_TP_PET_v2.csv")   # name, year, month, P, temp, PET(Thornthwaite) [kolonları kontrol et]
print('TP kolonları:', tp.columns.tolist())
si = pd.read_excel('/mnt/user-data/uploads/Paper 3/data/istasyon bilgileri.xlsx')
si.columns = ['name','lat_dms','lon_dms','elev']
def dms2dd(s):
    d,m,sec,h = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", s.strip()).groups()
    return (float(d)+float(m)/60+float(sec)/3600)
si['lat'] = si.lat_dms.map(dms2dd)
NAME_MAP = {'Corlu':'Çorlu','Edirne':'Edirne','Ipsala':'İpsala','Kirklareli':'Kırklareli',
            'Luleburgaz':'Lüleburgaz','Sariyer':'Sarıyer','Tekirdag':'Tekirdağ','Uzunkopru':'Uzunköprü'}
si['name'] = si['name'].map(NAME_MAP)
LAT = dict(zip(si.name, si.lat))

# ---- Oudin PET (aylık) ----
DIM = {1:31,2:28.25,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
MIDDOY = {1:15,2:46,3:75,4:105,5:136,6:166,7:197,8:228,9:258,10:289,11:319,12:350}
def ra_mj(lat_deg, m):
    phi = np.radians(lat_deg); J = MIDDOY[m]
    dr = 1 + 0.033*np.cos(2*np.pi*J/365)
    dec = 0.409*np.sin(2*np.pi*J/365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi)*np.tan(dec), -1, 1))
    return (24*60/np.pi)*0.0820*dr*(ws*np.sin(phi)*np.sin(dec) + np.cos(phi)*np.cos(dec)*np.sin(ws))
def oudin_month(T, lat_deg, m):
    Ra = ra_mj(lat_deg, m)
    pe_d = 0.408*Ra*(T+5)/100.0 if (T+5) > 0 else 0.0   # mm/gün (0.408=1/2.45)
    return max(pe_d, 0.0)*DIM[m]

tp['PET_oudin'] = [oudin_month(t, LAT[n], m) for n,t,m in zip(tp.name, tp.temp, tp.month)]
pcol = 'precip'
tp['B_th'] = tp[pcol] - tp['PET']
tp['B_ou'] = tp[pcol] - tp['PET_oudin']
print('PET yıllık ort (mm): Thornthwaite', tp.groupby('name').PET.sum().mean()/60 if False else round(tp.PET.mean()*12,0),
      '| Oudin', round(tp.PET_oudin.mean()*12,0))

# ---- SPEI (log-logistic PWM + fisk yedeği) — pipeline ile aynı ----
def spei_ll_pwm(x):
    x = np.asarray(x, float); v = np.isfinite(x); xv = x[v]
    n = len(xv)
    if n < 20: return None
    xs = np.sort(xv)
    F = (np.arange(1, n+1) - 0.35)/n
    b0 = xs.mean(); b1 = np.sum(F*xs)/n; b2 = np.sum(F**2*xs)/n
    g1 = 2*b1 - b0; g2 = 6*b2 - 6*b1 + b0
    beta = g1 and (2*g1 - b0)/(6*b1 - b0 - 6*b2) if (6*b1 - b0 - 6*b2)!=0 else None
    try:
        beta = (2*g1 - b0)/(6*b1 - b0 - 6*b2)
        alfa = (b0 - 2*b1)*beta/( (1+beta)*(beta) ) if False else None
    except Exception: pass
    # Vicente-Serrano 2010 3-par log-lojistik PWM (Singh-Maddala biçimi)
    w0, w1, w2 = b0, b1, b2
    beta = (2*w1 - w0)/(6*w1 - w0 - 6*w2)
    if not np.isfinite(beta) or beta <= 0: return None
    from math import gamma as G
    try:
        alfa = (w0 - 2*w1)*beta/(G(1+1/beta)*G(1-1/beta))
        gama = w0 - alfa*G(1+1/beta)*G(1-1/beta)
    except Exception: return None
    z = np.full_like(x, np.nan)
    F = np.clip(1.0/(1.0 + (alfa/np.maximum(x[v]-gama, 1e-9))**beta), 1e-6, 1-1e-6)
    out = np.full_like(x, np.nan); out[v] = stats.norm.ppf(F)
    return out
def spei_robust(x):
    x = np.asarray(x, float); v = np.isfinite(x)
    r = spei_ll_pwm(x)
    if r is not None and np.nanstd(r) > 0.5: return r
    out = np.full_like(x, np.nan); xv = x[v]
    try:
        c, loc, sc = stats.fisk.fit(xv - xv.min() + 1.0, floc=0)
        F = np.clip(stats.fisk.cdf(xv - xv.min() + 1.0, c, loc=0, scale=sc), 1e-6, 1-1e-6)
        out[v] = stats.norm.ppf(F); return out
    except Exception:
        return None

def build_spei(col, k=12):
    out = []
    for nm, g in tp.groupby('name'):
        g = g.sort_values(['year','month']).reset_index(drop=True)
        acc = g[col].rolling(k).sum().values
        z = np.full(len(g), np.nan)
        for m in range(1, 13):
            idx = (g.month==m).values & np.isfinite(acc)
            if idx.sum() < 20: continue
            zz = spei_robust(acc[idx])
            if zz is None: continue
            z[idx] = np.clip(zz, -4.75, 4.75)
        gg = g[['name','year','month']].copy(); gg['val'] = z
        out.append(gg)
    return pd.concat(out)

sp = pd.read_csv(W + "trakya_spi_spei_v2.csv")
th12 = sp[['name','year','month','SPEI12','SPI12']].copy()
ou = build_spei('B_ou', 12).rename(columns={'val':'SPEI12_ou'})
m = th12.merge(ou, on=['name','year','month'])
m = m.dropna(subset=['SPEI12','SPEI12_ou'])

# 1) korelasyon
cors = m.groupby('name').apply(lambda g: np.corrcoef(g.SPEI12, g.SPEI12_ou)[0,1])
print('\nSPEI12 Th-Oudin korelasyonu: medyan %.3f, aralık %.3f-%.3f' % (cors.median(), cors.min(), cors.max()))

# 2) dekadal ıraksama
m['dekad'] = (m.year//10)*10
d1 = m.groupby('dekad').apply(lambda g: (g.SPEI12-g.SPI12).mean()).round(3)
d2 = m.groupby('dekad').apply(lambda g: (g.SPEI12_ou-g.SPI12).mean()).round(3)
print('\nDekadal SPEI−SPI: Thornthwaite vs Oudin')
for dd in d1.index: print(f'  {dd}s: {d1[dd]:+.2f}  |  {d2[dd]:+.2f}')

# 3) ekstrem ay sayıları
for esik, ad in [(-1.0,'≤-1'), (-1.5,'≤-1.5'), (-2.0,'≤-2')]:
    a=(m.SPI12<=esik).sum(); b=(m.SPEI12<=esik).sum(); c=(m.SPEI12_ou<=esik).sum()
    print(f'{ad}: SPI={a}  SPEI-Th={b} ({(b-a)/max(a,1)*100:+.0f}%)  SPEI-Ou={c} ({(c-a)/max(a,1)*100:+.0f}%)')

# 4) D=12 kritik şiddet serileri (takvim yılı içi en kötü 12-ay penceresi = yıl içi 12 aylık tek pencere)
#    k=12, D=12 için pencere = Aralık'ta biten 12-aylık birikimin açığı ~ yıllık kritik şiddet
#    Mevcut pipeline tanımıyla uyum için: yıl içinde SPEI12'nin negatif kısmının toplamı DEĞİL;
#    kritik şiddet = yıl içindeki D=12 penceresinde Σmax(0, -SPEI12) benzeri.
#    Basit ve orijinalle birebir karşılaştırma: her yıl için S = Σ_ay max(0, -SPEI12) (12 ay tam pencere).
def crit12(g, col):
    s = np.maximum(0, -g[col].values)
    return s.sum() if np.isfinite(g[col].values).all() else np.nan
res = []
for nm, g in m.groupby('name'):
    for yr, gg in g.groupby('year'):
        if len(gg)==12:
            res.append({'name':nm,'year':yr,
                        'S_th':np.maximum(0,-gg.SPEI12.values).sum(),
                        'S_ou':np.maximum(0,-gg.SPEI12_ou.values).sum()})
cr = pd.DataFrame(res)
mkz = []
for nm, g in cr.groupby('name'):
    g = g.sort_values('year')
    for col in ['S_th','S_ou']:
        x = g[col].values
        n=len(x); s=0
        for i in range(n-1): s += np.sign(x[i+1:]-x[i]).sum()
        vals,cnt=np.unique(x,return_counts=True)
        v=n*(n-1)*(2*n+5)/18 - np.sum(cnt*(cnt-1)*(2*cnt+5))/18
        z=(s-np.sign(s))/np.sqrt(v) if s!=0 else 0
        mkz.append({'name':nm,'sürüm':col,'z':round(float(z),2)})
mk = pd.DataFrame(mkz).pivot(index='name', columns='sürüm', values='z')
print('\nD=12-benzeri yıllık şiddet MK z (Th vs Oudin):')
print(mk.to_string())
print('işaret uyumu:', (np.sign(mk.S_th)==np.sign(mk.S_ou)).mean())

m.to_csv(W+'rev_spei_oudin12.csv', index=False)
print('OK S4')
