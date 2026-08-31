# NS sentez figürü: (A) efektif S100(t) evrimi (Tekirdağ & İpsala, SPEI D=12)
#                   (B) S100 1980→2024 değişimi, tüm istasyonlar (SPEI D=12)
import pandas as pd, numpy as np, warnings
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

W = "/home/claude/work/"
cs = pd.read_csv(W + "kritik_siddet_serileri_tum.csv")
ns = pd.read_csv(W + "nsgev_sonuclari.csv")
YR_C, YR_S = 1995.0, 29.0

def gev_nll(theta, x, tt, nsflag):
    b0, b1, lsig, xi = theta if nsflag else (theta[0], 0.0, theta[1], theta[2])
    mu = b0 + b1*tt; sig = np.exp(lsig)
    z = (x - mu)/sig
    if abs(xi) < 1e-6: return float(np.sum(np.log(sig) + z + np.exp(-z)))
    w = 1 + xi*z
    if np.any(w <= 1e-10): return 1e10
    return float(np.sum(np.log(sig) + (1+1/xi)*np.log(w) + w**(-1/xi)))

def fit_ns(x, tt):
    c0, loc0, sc0 = stats.genextreme.fit(x)
    th0 = [loc0, 0.0, np.log(sc0), np.clip(-c0, -0.4, 0.4)]
    best = None
    for pert in [0.0, 0.1, -0.1]:
        t0 = list(th0); t0[0] *= (1+pert)
        r = optimize.minimize(gev_nll, t0, args=(x, tt, True), method='Nelder-Mead',
                              options={'maxiter':4000})
        if best is None or r.fun < best.fun: best = r
    return best.x

def fit_logistic(zero, tt):
    p = min(max(zero.mean(), 1e-6), 1-1e-6)
    def nll(th):
        pr = np.clip(1/(1+np.exp(-(th[0]+th[1]*tt))), 1e-9, 1-1e-9)
        return -np.sum(zero*np.log(pr) + (1-zero)*np.log(1-pr))
    return optimize.minimize(nll, [np.log(p/(1-p)), 0.0], method='Nelder-Mead').x

def gev_q(F, mu, sig, xi):
    if abs(xi) < 1e-6: return mu - sig*np.log(-np.log(F))
    return mu + sig/xi*((-np.log(F))**(-xi) - 1)

def s_T(T, mu, sig, xi, p0):
    pe = 1.0/T
    if pe >= (1-p0): return 0.0
    return max(gev_q(1 - pe/(1-p0), mu, sig, xi), 0.0)

INK, MUTE, C1, C2 = '#1F2937', '#9CA3AF', '#2563EB', '#C0392B'
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

# ---- Panel A ----
ax = axes[0]
years = np.arange(1966, 2025)
for st, c in [('Tekirdağ', C2), ('İpsala', C1)]:
    g = cs[(cs.name==st)&(cs.indeks=='SPEI')&(cs.k==12)&(cs.D==12)].sort_values('year')
    y = g.S.values; yr = g.year.values.astype(float)
    tt_all = (yr - YR_C)/YR_S
    nz = y > 0
    b0, b1, lsig, xi = fit_ns(y[nz], tt_all[nz]); sig = np.exp(lsig)
    a0, a1 = fit_logistic((y==0).astype(float), tt_all)
    ttp = (years - YR_C)/YR_S
    p0t = 1/(1+np.exp(-(a0 + a1*ttp)))
    s100 = [s_T(100, b0+b1*t, sig, xi, p) for t, p in zip(ttp, p0t)]
    ax.plot(years, s100, color=c, lw=2.2, label=f'{st} — NS model')
    # durağan referans
    from scipy.stats import genextreme as gev0
    c0, l0, s0 = gev0.fit(y[nz]); p0c = (y==0).mean()
    sstat = s_T(100, l0, s0, -c0, p0c)
    ax.axhline(sstat, color=c, lw=1.2, ls='--', alpha=0.6)
ax.set_title('A — 100 yıllık kuraklık şiddetinin zamanla evrimi (SPEI-12, D=12)', fontsize=10.5, loc='left')
ax.set_xlabel('Yıl'); ax.set_ylabel('S₁₀₀ (efektif 100-yıllık şiddet)')
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
h.append(Line2D([0],[0], color=MUTE, lw=1.2, ls='--')); l.append('durağan model referansı')
ax.legend(h, l, frameon=False, fontsize=9, loc='upper left')
ax.grid(axis='y', color='#E5E7EB', lw=0.6); ax.set_axisbelow(True)
for s_ in ['top','right']: ax.spines[s_].set_visible(False)

# ---- Panel B ----
ax = axes[1]
sub = ns[(ns.indeks=='SPEI')&(ns.D==12)].copy().sort_values('S100_2024')
ypos = np.arange(len(sub))
for i, (_, r) in enumerate(sub.iterrows()):
    sig_ = r.GEV_LRT_p < 0.05
    ax.plot([r.S100_1980, r.S100_2024], [i, i], color=MUTE, lw=1.6, zorder=1)
    ax.scatter([r.S100_1980], [i], color=MUTE, s=42, zorder=2, label='1980 iklimi' if i==0 else None)
    ax.scatter([r.S100_2024], [i], color=C2, s=52, zorder=3,
               edgecolor='white', linewidth=0.8, marker='o' if sig_ else 'D',
               label='2024 iklimi' if i==0 else None)
labels = [n + (' *' if n=='Kırklareli' else '') for n in sub.name]
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
ax.set_title('B — S₁₀₀: 1980 → 2024 iklimi (SPEI-12, D=12; ◆ = LRT p≥0.05)', fontsize=10.5, loc='left')
ax.set_xlabel('S₁₀₀ (100 yıllık kuraklık şiddeti)')
ax.annotate('* Kırklareli: ekstremler kayıt ortasında (2000–02); MK trendi anlamsız, NS eğimi kırılgan', xy=(0.01, -0.16), xycoords='axes fraction', fontsize=7.5, color='#6B7280')
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.grid(axis='x', color='#E5E7EB', lw=0.6); ax.set_axisbelow(True)
for s_ in ['top','right']: ax.spines[s_].set_visible(False)

plt.tight_layout()
plt.savefig(W + "ns_sdf_sentez.png", dpi=200, bbox_inches='tight')
print("OK: ns_sdf_sentez.png")
