# F10: Bileşik kuraklık risk haritası + bileşen ısı tablosu
import pandas as pd, numpy as np, re, shapefile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

W = "/home/claude/work/"
D3 = "/mnt/user-data/uploads/Paper 3/data/"
SHP = "/mnt/user-data/uploads/paper 2/files (60)/thrace-drought-scale-selection/thrace-drought-scale-selection/data/spatial/Trakya_Merged.shp"

tab = pd.read_csv(W + "risk_endeksi.csv", index_col=0)
NAME_MAP = {'Corlu':'Çorlu','Edirne':'Edirne','Ipsala':'İpsala','Kirklareli':'Kırklareli',
            'Luleburgaz':'Lüleburgaz','Sariyer':'Sarıyer','Tekirdag':'Tekirdağ','Uzunkopru':'Uzunköprü'}
si = pd.read_excel(D3 + "istasyon bilgileri.xlsx"); si.columns = ['name','lat_dms','lon_dms','elev']
def dms2dd(s):
    d,m,sec,h = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", s.strip()).groups()
    return (float(d)+float(m)/60+float(sec)/3600)*(-1 if h in ('S','W') else 1)
si['lat'] = si.lat_dms.map(dms2dd); si['lon'] = si.lon_dms.map(dms2dd)
si['name'] = si['name'].map(NAME_MAP)
tab = tab.merge(si[['name','lat','lon']], left_index=True, right_on='name')

fig = plt.figure(figsize=(13.5, 5.6))
gs_ = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])

# ---- Panel A: harita ----
ax = fig.add_subplot(gs_[0])
sf = shapefile.Reader(SHP)
segs = []
for shp in sf.shapes():
    pts = np.array(shp.points)
    parts = list(shp.parts) + [len(pts)]
    for a, b in zip(parts[:-1], parts[1:]):
        segs.append(pts[a:b])
ax.add_collection(LineCollection(segs, colors='#9CA3AF', linewidths=0.7))
cmap = plt.get_cmap('Reds')
sc = ax.scatter(tab.lon, tab.lat, c=tab.BILESIK, cmap=cmap, vmin=0, vmax=max(0.7, tab.BILESIK.max()),
                s=380, edgecolor='#1F2937', linewidth=1.2, zorder=5)
for _, r in tab.iterrows():
    ax.annotate(str(int(r.SIRA)), (r.lon, r.lat), ha='center', va='center', zorder=6,
                fontsize=10, fontweight='bold', color='white' if r.BILESIK > 0.35 else '#1F2937')
    lbl = r['name'] + (' *' if r['name']=='Lüleburgaz' else '')
    ax.annotate(lbl, (r.lon, r.lat), xytext=(0, -16), textcoords='offset points',
                ha='center', fontsize=8.5, color='#1F2937')
cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cb.set_label('Composite hazard-priority index (0–1)', fontsize=9); cb.ax.tick_params(labelsize=8)
ax.set_xlim(25.8, 29.4); ax.set_ylim(40.2, 42.3)
ax.set_title('A — Composite drought hazard-priority map (number in circle = rank)', fontsize=10.5, loc='left')
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.set_aspect(1/np.cos(np.radians(41.2)))
for s_ in ['top','right']: ax.spines[s_].set_visible(False)
ax.annotate('* Lüleburgaz: suspected temperature inhomogeneity — component C1 read with caution', xy=(0.01, -0.13),
            xycoords='axes fraction', fontsize=7.5, color='#6B7280')

# ---- Panel B: bileşen ısı tablosu ----
ax = fig.add_subplot(gs_[1])
comp_cols = ['C1_tarihsel_T50_n','C2_MK_z_n','C3_siklasma_n','C4_gelecek_dT50_n']
labels = ['C1\nHistorical\nseverity','C2\nIntensification\n(MK)','C3\nOccurrence\n(P₀ slope)','C4\nFuture\nΔT50']
M = tab.sort_values('BILESIK', ascending=True)
im = ax.pcolormesh(np.arange(5), np.arange(len(M)+1), M[comp_cols].values,
                   cmap='Reds', vmin=0, vmax=1, edgecolors='white', linewidth=2)
for i in range(len(M)):
    for j in range(4):
        v = M[comp_cols].values[i, j]
        ax.text(j+0.5, i+0.5, f'{v:.2f}', ha='center', va='center', fontsize=8.5,
                color='white' if v > 0.55 else '#1F2937')
ax.set_yticks(np.arange(len(M))+0.5)
ax.set_yticklabels([f"{n}  (#{int(s)})" for n, s in zip(M['name'], M.SIRA)], fontsize=9)
ax.set_xticks(np.arange(4)+0.5); ax.set_xticklabels(labels, fontsize=8)
ax.set_title('B — Normalized components (0 = lowest, 1 = highest risk)', fontsize=10.5, loc='left')
for s_ in ['top','right','left','bottom']: ax.spines[s_].set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
plt.savefig(W + "figs_EN/Fig8_risk_map.png", dpi=300, bbox_inches='tight')
print("OK: risk_haritasi_F10.png")
