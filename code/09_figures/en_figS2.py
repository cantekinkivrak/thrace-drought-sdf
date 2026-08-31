import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W = "/home/claude/work/"
master = pd.read_csv(W + "trakya_temp_filled_v2.csv")
# ---------- 3) QC figürü ----------
INK, MUTE, OLD, NEW = '#1F2937', '#9CA3AF', '#C0392B', '#2563EB'
tp = pd.read_csv("/mnt/user-data/uploads/paper 2/files (60)/thrace-drought-scale-selection/thrace-drought-scale-selection/data/derived/trakya_TP_PET_1965_2024.csv")

def series(dfin, st, y0, y1, col='temp'):
    s = dfin[(dfin.name==st)&(dfin.year>=y0)&(dfin.year<=y1)].sort_values(['year','month'])
    t = s.year + (s.month-0.5)/12
    return t.values, s[col].values, s

fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2))
plt.rcParams['font.family'] = 'DejaVu Sans'

# Panel A — İpsala 1975–1979
ax = axes[0]
t_tp, v_tp, _ = series(tp, 'İpsala', 1975, 1979)
t_new, v_new, s_new = series(master, 'İpsala', 1975, 1979)
nb = tp[(tp.name!='İpsala')&(tp.year>=1975)&(tp.year<=1979)].groupby(['year','month'])['temp'].mean().reset_index()
t_nb = nb.year + (nb.month-0.5)/12
gapA = ((s_new.year==1976)&(s_new.month.between(4,10))) | ((s_new.year==1978)&(s_new.month>=9))
obs = np.where(gapA.values, np.nan, v_new)
ax.plot(t_nb, nb.temp, color=MUTE, lw=1.2, label='Neighbouring-station mean')
ax.plot(t_tp, v_tp, color=OLD, lw=1.6, ls='--', label='Previous series (PCHIP infill)')
ax.plot(t_new, obs, color=INK, lw=1.8, label='Observations')
ax.plot(t_new[gapA.values], v_new[gapA.values], color=NEW, lw=0, marker='o', ms=5, label='New infill (v2)')
for y0, y1 in [(1976+3/12, 1976+10/12), (1978+8/12, 1979.0)]:
    ax.axvspan(y0, y1, color='#F3F4F6', zorder=0)
ax.annotate('Apr–Oct 1976: PCHIP assigns 4–10 °C to summer', xy=(1976.55, 8.5), fontsize=9, color=OLD)
ax.set_title('A — İpsala: infilling of missing months (previous PCHIP vs new neighbour-anomaly)', fontsize=11, loc='left')
ax.set_ylabel('Monthly mean temperature (°C)')
ax.legend(loc='upper right', fontsize=8.5, frameon=False, ncol=2)
ax.grid(axis='y', color='#E5E7EB', lw=0.6); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)

# Panel B — Uzunköprü 1965–1967
ax = axes[1]
t_tp, v_tp, _ = series(tp, 'Uzunköprü', 1965, 1967)
t_new, v_new, s_new = series(master, 'Uzunköprü', 1965, 1967)
nb = tp[(tp.name!='Uzunköprü')&(tp.year>=1965)&(tp.year<=1967)].groupby(['year','month'])['temp'].mean().reset_index()
t_nb = nb.year + (nb.month-0.5)/12
gapB = (s_new.year==1965)&(s_new.month.isin([1,2]))
obs = np.where(gapB.values, np.nan, v_new)
ax.plot(t_nb, nb.temp, color=MUTE, lw=1.2, label='Neighbouring-station mean')
ax.plot(t_tp, v_tp, color=OLD, lw=1.6, ls='--', label='Raw archive (incl. erroneous values)')
ax.plot(t_new, obs, color=INK, lw=1.8, label='Observations')
ax.plot(t_new[gapB.values], v_new[gapB.values], color=NEW, lw=0, marker='o', ms=6, label='Correction (v2)')
ax.scatter(t_tp[:2], v_tp[:2], color=OLD, marker='x', s=55, zorder=5)
ax.annotate('Jan–Feb 1965: −4.6 / −7.1 °C\n(neighbours +0.2…+5.8 °C)', xy=(1965.05, -6.5), fontsize=9, color=OLD)
ax.axvspan(1965.0, 1965+2/12, color='#F3F4F6', zorder=0)
ax.set_title('B — Uzunköprü: correction of erroneous source values', fontsize=11, loc='left')
ax.set_ylabel('Monthly mean temperature (°C)'); ax.set_xlabel('Year')
ax.legend(loc='upper right', fontsize=8.5, frameon=False, ncol=2)
ax.grid(axis='y', color='#E5E7EB', lw=0.6); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig(W + "figs_EN/FigS2_temperature_QC.png", dpi=300, bbox_inches='tight')
print("OK: FigS2")
