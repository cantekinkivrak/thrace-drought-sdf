# Adım 6: Copula tabanlı ortak şiddet-süre analizi (olay bazlı, k=12)
# T_AND = mu / P(S>s ve D>d), T_OR = mu / P(S>s veya D>d); mu = ort. olaylar-arası süre (yıl)
import pandas as pd, numpy as np, warnings
from scipy import stats, optimize, integrate
warnings.filterwarnings('ignore')

W = "/home/claude/work/"
ev = pd.read_csv(W + "olay_katalogu_tum.csv")
ev = ev[ev.k==12]
REC_YRS = 59.1   # Ara 1965 - Ara 2024

MARG = {'Gamma':stats.gamma, 'Weibull':stats.weibull_min, 'LN':stats.lognorm,
        'GEV':stats.genextreme, 'Ussel':stats.expon}

def ad_stat(x, dist, params):
    x = np.sort(x); n = len(x)
    F = np.clip(dist.cdf(x, *params), 1e-9, 1-1e-9)
    i = np.arange(1, n+1)
    return float(-n - np.mean((2*i-1)*(np.log(F) + np.log(1-F[::-1]))))

def best_marginal(x):
    out = []
    for nm, dist in MARG.items():
        try:
            p = dist.fit(x)
            a = ad_stat(x, dist, p)
            if np.isfinite(a): out.append((nm, dist, p, a))
        except Exception: pass
    out.sort(key=lambda t: t[3])
    return out[0]

# ---- Copula CDF'leri ----
def C_gumbel(u, v, th):
    return np.exp(-(((-np.log(u))**th + (-np.log(v))**th)**(1/th)))
def C_clayton(u, v, th):
    return np.maximum(u**(-th) + v**(-th) - 1, 1e-12)**(-1/th)
def C_frank(u, v, th):
    return -1/th*np.log(1 + (np.expm1(-th*u)*np.expm1(-th*v))/np.expm1(-th))
def C_joe(u, v, th):
    a = (1-u)**th; b = (1-v)**th
    return 1 - (a + b - a*b)**(1/th)
def C_gauss(u, v, rho):
    from scipy.stats import multivariate_normal, norm
    x = norm.ppf(u); y = norm.ppf(v)
    mvn = multivariate_normal(mean=[0,0], cov=[[1,rho],[rho,1]])
    pts = np.column_stack([np.atleast_1d(x).ravel(), np.atleast_1d(y).ravel()])
    c = mvn.cdf(pts)
    return c.reshape(np.atleast_1d(u).shape) if np.ndim(u) else float(c)

# ---- tau -> parametre ----
def th_gumbel(tau):  return 1/(1-tau)
def th_clayton(tau): return 2*tau/(1-tau)
def th_frank(tau):
    def f(th):
        D1 = integrate.quad(lambda t: t/np.expm1(t), 0, th)[0]/th
        return 1 - 4/th*(1 - D1) - tau
    return optimize.brentq(f, 0.05, 80)
def th_joe(tau):
    def tau_joe(th):
        s = sum(1/(k*(th*k+2)*(th*(k-1)+2)) for k in range(1, 3000))
        return 1 - 4*s
    return optimize.brentq(lambda th: tau_joe(th) - tau, 1.001, 60)
def th_gauss(tau):   return np.sin(np.pi*tau/2)

COPS = {'Gumbel':(C_gumbel, th_gumbel), 'Clayton':(C_clayton, th_clayton),
        'Frank':(C_frank, th_frank), 'Joe':(C_joe, th_joe), 'Gauss':(C_gauss, th_gauss)}
LAM_U = {'Gumbel': lambda th: 2 - 2**(1/th), 'Joe': lambda th: 2 - 2**(1/th),
         'Clayton': lambda th: 0.0, 'Frank': lambda th: 0.0, 'Gauss': lambda th: 0.0}

def emp_copula(u, v):
    n = len(u)
    return np.array([np.mean((u <= u[i]) & (v <= v[i])) for i in range(n)])

results, models = [], {}
for (st, fam), g in ev.groupby(['name','indeks']):
    S = g.S.values.astype(float); Dm = g.L_ay.values.astype(float)
    n = len(S); mu = REC_YRS/n
    tau = stats.kendalltau(S, Dm).statistic
    u = stats.rankdata(S)/(n+1); v = stats.rankdata(Dm)/(n+1)
    Ce = emp_copula(u, v)
    best = None
    for nm, (Cf, thf) in COPS.items():
        try:
            th = thf(tau)
            Cth = Cf(u, v, th)
            cvm = float(np.sum((Ce - Cth)**2))
            if best is None or cvm < best[3]: best = (nm, Cf, th, cvm)
        except Exception: pass
    nmS, dS, pS, adS = best_marginal(S)
    nmD, dD, pD, adD = best_marginal(Dm)
    cop_nm, Cf, th, cvm = best
    results.append({'name':st,'indeks':fam,'n_olay':n,'mu_yil':round(mu,2),
                    'Kendall_tau':round(tau,3),'copula':cop_nm,'theta':round(th,3),
                    'lambda_U':round(LAM_U[cop_nm](th),3),'CvM':round(cvm,4),
                    'S_marjinal':nmS,'S_AD':round(adS,3),'D_marjinal':nmD,'D_AD':round(adD,3)})
    models[(st,fam)] = dict(Cf=Cf, th=th, mu=mu, dS=dS, pS=pS, dD=dD, pD=pD, cop=cop_nm)

res = pd.DataFrame(results)
res.to_csv(W + "copula_fits.csv", index=False)
print("COPULA FİT ÖZETİ (k=12, olay bazlı):")
print(res.to_string(index=False))
print("\nSeçilen copulalar:", res.copula.value_counts().to_dict())

# ---- Rekor olayların ortak dönüş periyotları ----
def joint_T(st, fam, s, d):
    m = models[(st,fam)]
    FS = float(np.clip(m['dS'].cdf(s, *m['pS']), 0, 1-1e-9))
    FD = float(np.clip(m['dD'].cdf(d, *m['pD']), 0, 1-1e-9))
    C = float(m['Cf'](max(FS,1e-9), max(FD,1e-9), m['th']))
    p_and = max(1 - FS - FD + C, 1e-12)
    p_or  = max(1 - C, 1e-12)
    return m['mu']/p_and, m['mu']/p_or

rec_rows = []
for (st, fam), g in ev.groupby(['name','indeks']):
    r = g.loc[g.S.idxmax()]
    Tand, Tor = joint_T(st, fam, r.S, r.L_ay)
    rec_rows.append({'name':st,'indeks':fam,'olay':f"{int(r.bas_yil)}/{int(r.bas_ay):02d}–{int(r.son_yil)}/{int(r.son_ay):02d}",
                     'S':r.S,'D_ay':int(r.L_ay),
                     'T_AND_yil':round(min(Tand, 99999),0),'T_OR_yil':round(min(Tor, 99999),0)})
rec = pd.DataFrame(rec_rows)
rec.to_csv(W + "copula_rekor_olaylar.csv", index=False)
print("\nREKOR OLAYLARIN ORTAK DÖNÜŞ PERİYOTLARI:")
print(rec.sort_values(['indeks','T_AND_yil'], ascending=[True,False]).to_string(index=False))

import pickle
with open(W + "copula_models.pkl", "wb") as f: pickle.dump(models, f)
print("\nModeller kaydedildi (figür için).")
