# S1: Modified Mann-Kendall (Hamed-Rao 1998) + BH-FDR + alan anlamlılığı
import pandas as pd, numpy as np
from scipy import stats

W = "/home/claude/work/"
cs = pd.read_csv(W + "kritik_siddet_serileri_tum.csv")

def mk_stat(x):
    n = len(x)
    s = 0
    for i in range(n-1):
        s += np.sign(x[i+1:] - x[i]).sum()
    # bağ düzeltmeli varyans
    vals, cnt = np.unique(x, return_counts=True)
    v = n*(n-1)*(2*n+5)/18 - np.sum(cnt*(cnt-1)*(2*cnt+5))/18
    return s, v

def hamed_rao(x):
    """HR94: rank otokorelasyonuyla varyans düzeltmesi (anlamlı gecikmeler)."""
    n = len(x)
    s, v = mk_stat(x)
    if v <= 0: return np.nan, np.nan, np.nan
    r = stats.rankdata(x)
    rd = r - r.mean()
    denom = np.sum(rd**2)
    nsig = 0; corr = 0.0
    maxlag = min(n-2, n//3)
    for k in range(1, maxlag+1):
        rho = np.sum(rd[:-k]*rd[k:]) / denom
        if abs(rho) > 1.96/np.sqrt(n):   # anlamlı gecikmeler
            corr += (n-k)*(n-k-1)*(n-k-2)*rho
            nsig += 1
    nn = 1 + 2.0/(n*(n-1)*(n-2)) * corr
    nn = max(nn, 1e-6)
    v_c = v * nn
    z = (s - np.sign(s)) / np.sqrt(v_c) if s != 0 else 0.0
    p = 2*(1 - stats.norm.cdf(abs(z)))
    return z, p, nn

def bh(pvals, q=0.05):
    p = np.asarray(pvals); m = len(p)
    order = np.argsort(p); thresh = q*np.arange(1, m+1)/m
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    crit = p[order][k-1] if k > 0 else 0.0
    return (p <= crit), k

rows = []
for (nm, ix, k, D), g in cs.groupby(['name','indeks','k','D']):
    g = g.sort_values('year'); x = g.S.values.astype(float)
    if (x > 0).sum() < 10: continue
    z, p, nfac = hamed_rao(x)
    # orijinal (bağ düzeltmeli) MK
    s, v = mk_stat(x)
    z0 = (s - np.sign(s))/np.sqrt(v) if s != 0 else 0.0
    p0 = 2*(1 - stats.norm.cdf(abs(z0)))
    rows.append({'name':nm,'indeks':ix,'k':k,'D':D,'n':len(x),
                 'z_mk':round(z0,3),'p_mk':p0,'z_hr':round(z,3),'p_hr':p,'var_infl':round(nfac,3)})
t = pd.DataFrame(rows)
print('Toplam seri:', len(t))

for fam, sub in [('TÜMÜ', t), ('SPI', t[t.indeks=='SPI']), ('SPEI', t[t.indeks=='SPEI'])]:
    n = len(sub)
    a = (sub.p_mk < 0.05).sum(); b = (sub.p_hr < 0.05).sum()
    rej_mk, k_mk = bh(sub.p_mk.values); rej_hr, k_hr = bh(sub.p_hr.values)
    pos_hr_bh = ((sub.z_hr > 0).values & rej_hr).sum()
    print(f"{fam:5s} n={n:3d} | ham MK p<.05: {a} ({a/n*100:.0f}%) | HR p<.05: {b} ({b/n*100:.0f}%) | "
          f"BH-FDR(MK): {k_mk} | BH-FDR(HR): {k_hr} | BH-HR'de pozitif: {pos_hr_bh}")

# yön kontrolü: HR+BH sonrası anlamlı olup AZALAN var mı?
rej_hr_all, _ = bh(t.p_hr.values)
neg = t[rej_hr_all & (t.z_hr < 0)]
print('HR+BH sonrası anlamlı AZALAN seri:', len(neg))

# varyans şişmesi özeti
print('var_infl medyan/q90:', t.var_infl.median().round(2), t.var_infl.quantile(0.9).round(2))
t['sig_hr_bh'] = rej_hr_all
t.to_csv(W + 'rev_mk_hr_fdr.csv', index=False)

# NS-GEV ve P0 LRT'lerine BH
ns = pd.read_csv(W + 'nsgev_sonuclari.csv')
for col, ad in [('GEV_LRT_p','NS-GEV'), ('P0_LRT_p','P0-lojistik')]:
    for ix in ['SPI','SPEI']:
        sub = ns[ns.indeks==ix]
        rej, kk = bh(sub[col].values)
        print(f'{ad} {ix}: ham p<.05 = {(sub[col]<0.05).sum()}/{len(sub)} | BH-FDR = {kk}/{len(sub)}')
    rej_all, kall = bh(ns[col].values)
    print(f'{ad} TÜM: ham = {(ns[col]<0.05).sum()}/{len(ns)} | BH = {kall}/{len(ns)}')
ns['GEV_sig_bh'] = bh(ns.GEV_LRT_p.values)[0]
ns['P0_sig_bh'] = bh(ns.P0_LRT_p.values)[0]
ns.to_csv(W + 'rev_ns_bh.csv', index=False)
print('OK S1')
