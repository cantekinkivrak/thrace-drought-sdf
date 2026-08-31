# Adım 5: (a) MK+Sen trend taraması (tüm hücreler) (b) NS-GEV μ(t) + lojistik P0(t) (k=12)
import pandas as pd, numpy as np, warnings
from scipy import stats, optimize
warnings.filterwarnings('ignore')

W = "/home/claude/work/"
cs = pd.read_csv(W + "kritik_siddet_serileri_tum.csv")
YR_C, YR_S = 1995.0, 29.0          # zaman kovaryatı: tt = (yıl-1995)/29  → [-1, 1]

# ---------------- (a) Mann-Kendall + Sen eğimi ----------------
def mk_test(x):
    n = len(x)
    s = 0
    for i in range(n-1):
        s += np.sign(x[i+1:] - x[i]).sum()
    uq, cnt = np.unique(x, return_counts=True)
    var = (n*(n-1)*(2*n+5) - np.sum(cnt*(cnt-1)*(2*cnt+5))) / 18.0
    if var <= 0: return 0.0, 1.0
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return z, 2*(1 - stats.norm.cdf(abs(z)))

def sen_slope(y, t):
    sl = [(y[j]-y[i])/(t[j]-t[i]) for i in range(len(y)-1) for j in range(i+1, len(y)) if t[j]!=t[i]]
    return float(np.median(sl)) if sl else 0.0

rows = []
for (st, fam, k, D), g in cs.groupby(['name','indeks','k','D']):
    g = g.sort_values('year')
    y = g.S.values; t = g.year.values.astype(float)
    if (y > 0).sum() < 10: continue
    z, p = mk_test(y)
    rows.append({'name':st,'indeks':fam,'k':k,'D':D,'MK_z':round(z,2),'MK_p':round(p,4),
                 'Sen_egim_10yil':round(sen_slope(y, t)*10, 3)})
mk = pd.DataFrame(rows)
mk.to_csv(W + "mk_trend_sonuclari.csv", index=False)
sig = mk[mk.MK_p < 0.05]
print(f"MK taraması: {len(mk)} hücre; anlamlı (p<0.05): {len(sig)} (%{100*len(sig)/len(mk):.0f})")
print("Anlamlı hücrelerin dağılımı (indeks × k):")
print(sig.groupby(['indeks','k']).size().unstack(fill_value=0).to_string())
print("Anlamlı hücrelerde yön: artan =", (sig.MK_z>0).sum(), "| azalan =", (sig.MK_z<0).sum())

# ---------------- (b) NS-GEV (k=12) ----------------
def gev_nll(theta, x, tt, ns):
    b0, b1, lsig, xi = theta if ns else (theta[0], 0.0, theta[1], theta[2])
    mu = b0 + b1*tt; sig = np.exp(lsig)
    z = (x - mu)/sig
    if abs(xi) < 1e-6:
        return float(np.sum(np.log(sig) + z + np.exp(-z)))
    w = 1 + xi*z
    if np.any(w <= 1e-10): return 1e10
    return float(np.sum(np.log(sig) + (1+1/xi)*np.log(w) + w**(-1/xi)))

def fit_gev(x, tt, ns):
    c0, loc0, sc0 = stats.genextreme.fit(x)
    xi0 = -c0
    if ns:
        th0 = [loc0, 0.0, np.log(sc0), np.clip(xi0, -0.4, 0.4)]
    else:
        th0 = [loc0, np.log(sc0), np.clip(xi0, -0.4, 0.4)]
    best = None
    for pert in [0.0, 0.1, -0.1]:
        t0 = list(th0)
        t0[0] *= (1+pert)
        r = optimize.minimize(gev_nll, t0, args=(x, tt, ns), method='Nelder-Mead',
                              options={'maxiter':4000, 'xatol':1e-6, 'fatol':1e-6})
        if best is None or r.fun < best.fun: best = r
    return best

def gev_q(F, mu, sig, xi):
    if abs(xi) < 1e-6: return mu - sig*np.log(-np.log(F))
    return mu + sig/xi*((-np.log(F))**(-xi) - 1)

def logistic_p0(zero, tt):
    # sabit model
    p = zero.mean(); p = min(max(p, 1e-6), 1-1e-6)
    nll0 = -np.sum(zero*np.log(p) + (1-zero)*np.log(1-p))
    def nll(th):
        eta = th[0] + th[1]*tt
        pr = 1/(1+np.exp(-eta)); pr = np.clip(pr, 1e-9, 1-1e-9)
        return -np.sum(zero*np.log(pr) + (1-zero)*np.log(1-pr))
    r = optimize.minimize(nll, [np.log(p/(1-p)), 0.0], method='Nelder-Mead')
    lrt_p = 1 - stats.chi2.cdf(2*(nll0 - r.fun), 1)
    return r.x, lrt_p, p

T_SHOW = [50, 100]
res = []
for (st, fam), g0 in cs[(cs.k==12)].groupby(['name','indeks']):
    for D in [3, 6, 9, 12]:
        g = g0[g0.D==D].sort_values('year')
        y = g.S.values; yr = g.year.values.astype(float)
        tt_all = (yr - YR_C)/YR_S
        nzm = y > 0
        if nzm.sum() < 18: continue
        x = y[nzm]; tt = tt_all[nzm]
        # lojistik P0(t)
        (a0, a1), p0_lrt_p, p0_const = logistic_p0((y==0).astype(float), tt_all)
        # GEV: durağan vs NS
        rs = fit_gev(x, tt, ns=False); rn = fit_gev(x, tt, ns=True)
        lrt = 2*(rs.fun - rn.fun); lrt_p = 1 - stats.chi2.cdf(max(lrt, 0), 1)
        b0, b1, lsig, xi = rn.x; sig = np.exp(lsig)
        row = {'name':st,'indeks':fam,'D':D,'n_nonzero':int(nzm.sum()),
               'GEV_b1':round(b1,3),'GEV_LRT_p':round(lrt_p,4),
               'P0_sabit':round(p0_const,3),'P0_egim':round(a1,3),'P0_LRT_p':round(p0_lrt_p,4)}
        # efektif dönüş seviyeleri: 1980 ve 2024 iklimi (NS model; P0 anlamlıysa lojistik, değilse sabit)
        for yr_ref, tag in [(1980, '1980'), (2024, '2024')]:
            ttr = (yr_ref - YR_C)/YR_S
            p0r = 1/(1+np.exp(-(a0 + a1*ttr))) if p0_lrt_p < 0.05 else p0_const
            mur = b0 + b1*ttr
            for T in T_SHOW:
                pe = 1.0/T
                if pe >= (1-p0r): row[f'S{T}_{tag}'] = 0.0
                else:
                    F = 1 - pe/(1-p0r)
                    row[f'S{T}_{tag}'] = round(float(max(gev_q(F, mur, sig, xi), 0)), 2)
        # eşdeğer dönüş periyodu: 2024'ün 100-yıllık şiddeti, 1980 ikliminde kaç yıllık?
        s100_24 = row.get('S100_2024', 0)
        tt80 = (1980 - YR_C)/YR_S
        p080 = 1/(1+np.exp(-(a0 + a1*tt80))) if p0_lrt_p < 0.05 else p0_const
        mu80 = b0 + b1*tt80
        if s100_24 > 0:
            if abs(xi) < 1e-6: F80 = np.exp(-np.exp(-(s100_24-mu80)/sig))
            else:
                w = 1 + xi*(s100_24-mu80)/sig
                F80 = np.exp(-w**(-1/xi)) if w > 0 else 1.0
            pex = (1-p080)*(1-F80)
            row['T_esdeger_1980'] = round(1/pex, 0) if pex > 1e-6 else np.inf
        res.append(row)

ns = pd.DataFrame(res)
ns.to_csv(W + "nsgev_sonuclari.csv", index=False)
print(f"\nNS-GEV (k=12): {len(ns)} hücre fitlendi")
print("GEV μ(t) anlamlı (LRT p<0.05):", (ns.GEV_LRT_p<0.05).sum(), "| p<0.10:", (ns.GEV_LRT_p<0.10).sum())
print("P0(t) anlamlı (p<0.05):", (ns.P0_LRT_p<0.05).sum(), "| P0 eğimi negatif (kuraklık sıklaşıyor):", (ns.P0_egim<0).sum())
print("\nSPEI, D=12 özeti:")
sub = ns[(ns.indeks=='SPEI')&(ns.D==12)][['name','n_nonzero','GEV_b1','GEV_LRT_p','P0_egim','P0_LRT_p','S100_1980','S100_2024','T_esdeger_1980']]
print(sub.to_string(index=False))
print("\nSPI, D=12 özeti:")
sub2 = ns[(ns.indeks=='SPI')&(ns.D==12)][['name','n_nonzero','GEV_b1','GEV_LRT_p','P0_egim','P0_LRT_p','S100_1980','S100_2024','T_esdeger_1980']]
print(sub2.to_string(index=False))
