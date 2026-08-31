# F9: Gelecek SDF eğrileri (SPEI-12, T=50) — bölge medyanı, model bandı
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W = "/home/claude/work/"

gs = pd.read_csv(W + "gelecek_sdf_fits.csv")
hist = pd.read_csv(W + "sdf_fits_tum.csv")
INK, BLU, RED = '#1F2937', '#2563EB', '#C0392B'
DS = [3, 6, 9, 12]

hmed = hist[(hist.k==12)&(hist.indeks=='SPEI')&(hist.D.isin(DS))].groupby('D')['T50'].median()

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
for ax, wname in zip(axes, ['2030-2059', '2070-2099']):
    ax.plot(hmed.index, hmed.values, color=INK, lw=2.2, marker='o', ms=4, label='Tarihsel (1966–2024)')
    for ssp, c, lbl in [('ssp245', BLU, 'SSP2-4.5'), ('ssp585', RED, 'SSP5-8.5')]:
        sub = gs[(gs.ssp==ssp)&(gs.pencere==wname)&(gs.indeks=='SPEI')]
        permod = sub.groupby(['model','D'])['T50'].median().reset_index()
        med = permod.groupby('D')['T50'].median()
        lo = permod.groupby('D')['T50'].min(); hi = permod.groupby('D')['T50'].max()
        nmod = permod.model.nunique()
        ax.plot(med.index, med.values, color=c, lw=2.0, marker='o', ms=4,
                label=f'{lbl} (medyan, {nmod} model)')
        ax.fill_between(med.index, lo.reindex(med.index), hi.reindex(med.index), color=c, alpha=0.15)
    ax.set_title(f'{"A" if wname=="2030-2059" else "B"} — {wname}', fontsize=11, loc='left')
    ax.set_xlabel('Kuraklık süresi D (ay)'); ax.set_xticks(DS)
    ax.grid(axis='y', color='#E5E7EB', lw=0.6); ax.set_axisbelow(True)
    for s_ in ['top','right']: ax.spines[s_].set_visible(False)
axes[0].set_ylabel('Kritik kuraklık şiddeti S (T = 50 yıl)')
axes[0].legend(frameon=False, fontsize=9, loc='upper left')
fig.suptitle('Gelecek SDF eğrileri — SPEI-12, bölge medyanı (QDM-düzeltmeli CMIP6; bant: model aralığı)',
             fontsize=11.5, y=1.02)
plt.tight_layout()
plt.savefig(W + "cmip6_gelecek_sdf.png", dpi=200, bbox_inches='tight')
print("OK: cmip6_gelecek_sdf.png")
