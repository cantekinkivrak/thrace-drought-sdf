# S3: Copula sağlamlık paketi
#  (a) sınır-olayı (sağ/sol sansür) duyarlılığı — Tekirdağ & İpsala + tüm istasyon tau kayması
#  (b) parametrik bootstrap CvM GOF (16 fit, B=400)
#  (c) sabit-aile duyarlılığı (Gumbel / Frank zorlamalı) — rekor olay T_AND
#  (d) süre-normalize bağımlılık: tau(S/D, D)
import pandas as pd, numpy as np, warnings
from scipy import stats, optimize, integrate
warnings.filterwarnings('ignore')
W = "/home/claude/work/"
ev = pd.read_csv(W + "olay_katalogu_tum.csv"); ev = ev[ev.k==12]
REC_YRS = 59.1

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
            p = dist.fit(x); a = ad_stat(x, dist, p)
            if np.isfinite(a): out.append((nm, dist, p, a))
        except Exception: pass
    out.sort(key=lambda t: t[3]); return out[0]

def C_gumbel(u,v,th): return np.exp(-(((-np.log(u))**th + (-np.log(v))**th)**(1/th)))
def C_clayton(u,v,th): return np.maximum(u**(-th)+v**(-th)-1, 1e-12)**(-1/th)
def C_frank(u,v,th): return -1/th*np.log(1 + (np.expm1(-th*u)*np.expm1(-th*v))/np.expm1(-th))
def C_joe(u,v,th):
    a=(1-u)**th; b=(1-v)**th
    return 1-(a+b-a*b)**(1/th)
def C_gauss(u,v,rho):
    from scipy.stats import norm
    x=norm.ppf(u); y=norm.ppf(v)
    # vektörize edilebilir Owen yaklaşımı yerine scipy mvn (nokta bazlı)
    from scipy.stats import multivariate_normal
    mvn = multivariate_normal(mean=[0,0], cov=[[1,rho],[rho,1]])
    pts = np.column_stack([np.atleast_1d(x).ravel(), np.atleast_1d(y).ravel()])
    c = mvn.cdf(pts)
    return c.reshape(np.atleast_1d(u).shape) if np.ndim(u) else float(c)

def th_gumbel(tau): return 1/(1-tau)
def th_clayton(tau): return 2*tau/(1-tau)
def th_frank(tau):
    def f(th):
        D1 = integrate.quad(lambda t: t/np.expm1(t), 0, th)[0]/th
        return 1 - 4/th*(1-D1) - tau
    return optimize.brentq(f, 0.05, 120)
def th_joe(tau):
    ks = np.arange(1, 3000)
    def tau_joe(th): return 1 - 4*np.sum(1/(ks*(th*ks+2)*(th*(ks-1)+2)))
    return optimize.brentq(lambda th: tau_joe(th)-tau, 1.001, 80)
def th_gauss(tau): return np.sin(np.pi*tau/2)
COPS = {'Gumbel':(C_gumbel,th_gumbel),'Clayton':(C_clayton,th_clayton),
        'Frank':(C_frank,th_frank),'Joe':(C_joe,th_joe),'Gauss':(C_gauss,th_gauss)}

def emp_copula(u, v):
    n=len(u)
    return np.array([np.mean((u<=u[i])&(v<=v[i])) for i in range(n)])

def fit_copula(S, Dm):
    n=len(S)
    tau = stats.kendalltau(S, Dm).statistic
    u = stats.rankdata(S)/(n+1); v = stats.rankdata(Dm)/(n+1)
    Ce = emp_copula(u,v)
    best=None
    for nm,(Cf,thf) in COPS.items():
        try:
            th=thf(tau); cvm=float(np.sum((Ce-Cf(u,v,th))**2))
            if best is None or cvm<best[3]: best=(nm,Cf,th,cvm)
        except Exception: pass
    return tau, best  # (nm, Cf, th, cvm)

def t_and(S, Dm, s_star, d_star, best):
    n=len(S); mu=REC_YRS/n
    nmS,dS,pS,_ = best_marginal(S); nmD,dD,pD,_ = best_marginal(Dm)
    FS=float(np.clip(dS.cdf(s_star,*pS),1e-9,1-1e-9))
    FD=float(np.clip(dD.cdf(d_star,*pD),1e-9,1-1e-9))
    nm,Cf,th,_=best
    C=float(Cf(FS,FD,th))
    return mu/max(1-FS-FD+C,1e-12)

# ---------- (a) sınır olayı duyarlılığı ----------
print('=== (a) SINIR OLAYI DUYARLILIĞI ===')
recs = {('Tekirdağ','SPEI'):(90.016,72), ('Tekirdağ','SPI'):(70.735,55),
        ('İpsala','SPEI'):(92.731,72), ('İpsala','SPI'):(82.565,69)}
rows=[]
for (st,fam),(s_star,d_star) in recs.items():
    g = ev[(ev.name==st)&(ev.indeks==fam)]
    S=g.S.values.astype(float); Dm=g.L_ay.values.astype(float)
    term = (g.son_yil>=2024).values           # kayıt sonunda süren
    left = (g.bas_yil<=1966).values           # kayıt başına dokunan
    for etiket, mask in [('tam', np.ones(len(g),bool)),
                         ('son-olay hariç', ~term),
                         ('sınır olayları hariç', ~(term|left))]:
        Ss, Ds = S[mask], Dm[mask]
        tau, best = fit_copula(Ss, Ds)
        T = t_and(Ss, Ds, s_star, d_star, best)
        rows.append({'istasyon':st,'indeks':fam,'varyant':etiket,'n':int(mask.sum()),
                     'tau':round(tau,3),'copula':best[0],'T_AND_rekor':round(T,1)})
        print(f'{st:9s} {fam:4s} {etiket:22s} n={mask.sum():2d} tau={tau:.3f} {best[0]:8s} T_AND={T:.0f} yıl')
pd.DataFrame(rows).to_csv(W+'rev_copula_censoring.csv', index=False)

# tüm istasyonlarda tau kayması (son-olay hariç)
d_tau=[]
for (st,fam),g in ev.groupby(['name','indeks']):
    S=g.S.values.astype(float); Dm=g.L_ay.values.astype(float)
    t0=stats.kendalltau(S,Dm).statistic
    m=~(g.son_yil>=2024).values
    if m.sum()>=15 and m.sum()<len(g):
        t1=stats.kendalltau(S[m],Dm[m]).statistic
        d_tau.append(t1-t0)
print(f'Tau kayması (son-olay hariç, {len(d_tau)} fit): medyan {np.median(d_tau):+.3f}, aralık [{min(d_tau):+.3f},{max(d_tau):+.3f}]')

# ---------- (b) parametrik bootstrap CvM GOF ----------
print('\n=== (b) BOOTSTRAP CvM GOF (B=400) ===')
def cond_sample(nm, Cf, th, n, rng):
    if nm=='Gauss':
        rho=th
        z1=rng.standard_normal(n); z2=rho*z1+np.sqrt(1-rho**2)*rng.standard_normal(n)
        return stats.norm.cdf(z1), stats.norm.cdf(z2)
    u = rng.uniform(1e-4, 1-1e-4, n); w = rng.uniform(1e-4, 1-1e-4, n)
    h = 1e-5
    lo=np.full(n,1e-6); hi=np.full(n,1-1e-6)
    for _ in range(38):
        mid=(lo+hi)/2
        Cu=(Cf(np.clip(u+h,1e-9,1-1e-9),mid,th)-Cf(np.clip(u-h,1e-9,1-1e-9),mid,th))/(2*h)
        take = Cu < w
        lo=np.where(take, mid, lo); hi=np.where(take, hi, mid)
    return u, (lo+hi)/2

gof=[]
for (st,fam),g in ev.groupby(['name','indeks']):
    S=g.S.values.astype(float); Dm=g.L_ay.values.astype(float); n=len(S)
    tau, best = fit_copula(S,Dm)
    nm,Cf,th,cvm_obs = best
    rng=np.random.default_rng(7)
    cnt=0; ok=0
    for b in range(400):
        try:
            us,vs = cond_sample(nm,Cf,th,n,rng)
            taub = stats.kendalltau(us,vs).statistic
            thb = COPS[nm][1](max(min(taub,0.985),0.02))
            ub=stats.rankdata(us)/(n+1); vb=stats.rankdata(vs)/(n+1)
            cvb=float(np.sum((emp_copula(ub,vb)-Cf(ub,vb,thb))**2))
        except Exception: continue
        ok+=1
        if cvb>=cvm_obs: cnt+=1
    p = cnt/ok if ok>50 else np.nan
    gof.append({'name':st,'indeks':fam,'copula':nm,'CvM':round(cvm_obs,4),'p_boot':round(p,3),'B_ok':ok})
    print(f'{st:10s} {fam:4s} {nm:8s} CvM={cvm_obs:.4f} p_boot={p:.3f}')
gdf=pd.DataFrame(gof); gdf.to_csv(W+'rev_copula_gof.csv', index=False)
print('p>=0.05 geçen:', (gdf.p_boot>=0.05).sum(), '/', len(gdf))

# ---------- (c) sabit-aile duyarlılığı ----------
print('\n=== (c) SABİT AİLE (rekor olay T_AND) ===')
rows=[]
for (st,fam),(s_star,d_star) in recs.items():
    g=ev[(ev.name==st)&(ev.indeks==fam)]
    S=g.S.values.astype(float); Dm=g.L_ay.values.astype(float)
    tau,_=fit_copula(S,Dm)
    for nm in ['Gumbel','Frank','Joe','Gauss']:
        Cf,thf=COPS[nm]
        try:
            th=thf(tau)
            T=t_and(S,Dm,s_star,d_star,(nm,Cf,th,0))
            rows.append({'istasyon':st,'indeks':fam,'aile':nm,'T_AND':round(T,1)})
        except Exception: pass
sf=pd.DataFrame(rows)
for (st,fam),g in sf.groupby(['istasyon','indeks']):
    print(f'{st:10s} {fam:4s}: ' + ', '.join(f'{r.aile}={r.T_AND:.0f}' for _,r in g.iterrows()))
sf.to_csv(W+'rev_copula_fixedfam.csv', index=False)

# ---------- (d) yoğunluk-temelli bağımlılık ----------
print('\n=== (d) tau(S/D, D) — mekanik birikimden arındırılmış ===')
rows=[]
for (st,fam),g in ev.groupby(['name','indeks']):
    S=g.S.values.astype(float); Dm=g.L_ay.values.astype(float)
    I=S/np.maximum(Dm,1)
    t_sd=stats.kendalltau(S,Dm).statistic
    t_id=stats.kendalltau(I,Dm).statistic
    rows.append({'name':st,'indeks':fam,'tau_S_D':round(t_sd,3),'tau_I_D':round(t_id,3)})
idf=pd.DataFrame(rows); idf.to_csv(W+'rev_copula_intensity.csv', index=False)
print('tau(S,D) medyan:', idf.tau_S_D.median().round(3), '| tau(S/D, D) medyan:', idf.tau_I_D.median().round(3),
      '| aralık:', idf.tau_I_D.min().round(2), '-', idf.tau_I_D.max().round(2))
print('OK S3')
