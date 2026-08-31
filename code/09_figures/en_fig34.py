# Sentez figürleri: (1) k=12 sekiz istasyon SPI vs SPEI SDF; (2) ölçek duyarlılığı (bölge medyanı, T=50)
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W = "/home/claude/work/"
f = pd.read_csv(W + "sdf_fits_tum.csv")
SPI_C, SPEI_C = '#2563EB', '#C0392B'
STS = ['Edirne','Kırklareli','Lüleburgaz','Uzunköprü','İpsala','Çorlu','Tekirdağ','Sarıyer']

# ---------- Figür 1: k=12, 8 istasyon, T=10 (kesikli) & T=100 (düz) ----------
fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True, sharey=True)
for ax, st in zip(axes.flat, STS):
    for fam, c in [('SPI', SPI_C), ('SPEI', SPEI_C)]:
        sub = f[(f.name==st)&(f.indeks==fam)&(f.k==12)].sort_values('D')
        ax.plot(sub.D, sub.T100, color=c, lw=1.8, marker='o', ms=3)
        ax.plot(sub.D, sub.T10,  color=c, lw=1.4, ls='--', marker='o', ms=2.5, alpha=0.75)
    ax.set_title(st, fontsize=10, loc='left')
    ax.grid(axis='y', color='#E5E7EB', lw=0.5); ax.set_axisbelow(True)
    ax.set_xticks([1,3,6,9,12])
    for s_ in ['top','right']: ax.spines[s_].set_visible(False)
for ax in axes[1]: ax.set_xlabel('Duration D (months)')
for ax in axes[:,0]: ax.set_ylabel('Severity S')
leg = [Line2D([0],[0], color=SPI_C, lw=1.8, label='SPI-12'),
       Line2D([0],[0], color=SPEI_C, lw=1.8, label='SPEI-12'),
       Line2D([0],[0], color='#6B7280', lw=1.8, label='T = 100 yr'),
       Line2D([0],[0], color='#6B7280', lw=1.4, ls='--', label='T = 10 yr')]
fig.legend(handles=leg, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
fig.suptitle('Stationary SDF curves — 12-month scale, 8 stations (1966–2024)', y=1.06, fontsize=12)
plt.tight_layout()
plt.savefig(W + "figs_EN/Fig3_sdf_8stations_k12.png", dpi=300, bbox_inches='tight')
print("OK: sdf_8istasyon_k12.png")

# ---------- Figür 2: ölçek duyarlılığı — bölge medyanı T=50 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
cmap = plt.get_cmap('Blues')
kcol = {k: cmap(0.35 + 0.6*i/4) for i, k in enumerate([1,3,6,9,12])}
for ax, fam in zip(axes, ['SPI','SPEI']):
    for k in [1,3,6,9,12]:
        sub = f[(f.indeks==fam)&(f.k==k)]
        med = sub.groupby('D')['T50'].median()
        ax.plot(med.index, med.values, color=kcol[k], lw=1.9, marker='o', ms=3.5, label=f'k = {k} month' + ('s' if k > 1 else ''))
    ax.set_title(f'{"A" if fam=="SPI" else "B"} — {fam} (regional median, T = 50 yr)', fontsize=11, loc='left')
    ax.set_xlabel('Duration D (months)'); ax.set_xticks([1,3,6,9,12])
    ax.grid(axis='y', color='#E5E7EB', lw=0.6); ax.set_axisbelow(True)
    for s_ in ['top','right']: ax.spines[s_].set_visible(False)
axes[0].set_ylabel('Severity S')
axes[1].legend(fontsize=9, frameon=False, loc='upper left', title='Accumulation scale')
plt.tight_layout()
plt.savefig(W + "figs_EN/Fig4_sdf_scale_sensitivity.png", dpi=300, bbox_inches='tight')
print("OK: sdf_olcek_duyarlilik.png")
