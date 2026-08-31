# Copula kontur figürü: Tekirdağ SPI-12 & SPEI-12 — T_AND ortak dönüş periyodu konturları
import pandas as pd, numpy as np, pickle, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

W = "/home/claude/work/"
ev = pd.read_csv(W + "olay_katalogu_tum.csv"); ev = ev[ev.k==12]

# pickle çözümü için copula CDF'leri (copula_step6 ile aynı isimler)
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

models = pickle.load(open(W + "copula_models.pkl", "rb"))

ST = 'Tekirdağ'
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
levels = [5, 10, 25, 50, 100]
cmap = plt.get_cmap('Blues')
lcols = [cmap(0.35 + 0.6*i/(len(levels)-1)) for i in range(len(levels))]

for ax, fam in zip(axes, ['SPI','SPEI']):
    m = models[(ST, fam)]
    g = ev[(ev.name==ST)&(ev.indeks==fam)]
    dd = np.linspace(1, 84, 170); ss = np.linspace(0.5, 110, 220)
    DD, SS = np.meshgrid(dd, ss)
    FD = np.clip(m['dD'].cdf(DD, *m['pD']), 1e-9, 1-1e-9)
    FS = np.clip(m['dS'].cdf(SS, *m['pS']), 1e-9, 1-1e-9)
    C = m['Cf'](FS, FD, m['th'])
    T_AND = m['mu'] / np.maximum(1 - FS - FD + C, 1e-12)
    cs_ = ax.contour(DD, SS, T_AND, levels=levels, colors=lcols, linewidths=1.8)
    ax.clabel(cs_, fmt=lambda v: f'{v:.0f} yr', fontsize=8)
    ax.scatter(g.L_ay, g.S, s=26, color='#9CA3AF', edgecolor='white', linewidth=0.6, zorder=3,
               label='Observed events (1966–2024)')
    r = g.loc[g.S.idxmax()]
    ax.scatter([r.L_ay], [r.S], marker='*', s=260, color='#C0392B', edgecolor='white', zorder=4,
               label=f"{int(r.bas_yil)}–{int(r.son_yil)} event")
for ax, fam in zip(axes, ['SPI','SPEI']):
    m = models[(ST, fam)]
    g = ev[(ev.name==ST)&(ev.indeks==fam)]
    r = g.loc[g.S.idxmax()]
    FS = float(np.clip(m['dS'].cdf(r.S, *m['pS']), 0, 1-1e-9))
    FD = float(np.clip(m['dD'].cdf(r.L_ay, *m['pD']), 0, 1-1e-9))
    C = float(m['Cf'](max(FS,1e-9), max(FD,1e-9), m['th']))
    T = m['mu']/max(1-FS-FD+C, 1e-12)
    ax.annotate(f"T_AND ≈ {T:.0f} yr\n(S = {r.S:.0f}, D = {int(r.L_ay)} months)",
                xy=(r.L_ay, r.S), xytext=(r.L_ay-30, r.S-1),
                fontsize=8.5, color='#C0392B')
    ax.set_title(f'{"A" if fam=="SPI" else "B"} — {ST}, {fam}-12  (copula: {m["cop"]})', fontsize=10.5, loc='left')
    ax.set_xlabel('Event duration D (months)')
    ax.grid(color='#EEF1F3', lw=0.5); ax.set_axisbelow(True)
    for s_ in ['top','right']: ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
axes[0].set_ylabel('Event severity S')
plt.tight_layout()
plt.savefig(W + "figs_EN/Fig6_copula_TAND_Tekirdag.png", dpi=300, bbox_inches='tight')
print("OK: tekirdag_copula_TAND.png")
