# Figure 1 — Study area map (English), Journal of Hydrology
import pandas as pd, numpy as np, re, json, shapefile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib.patheffects as pe

W = "/home/claude/work/"
SHP = "/mnt/user-data/uploads/paper 2/files (60)/thrace-drought-scale-selection/thrace-drought-scale-selection/data/spatial/Trakya_Merged.shp"

NAME_MAP = {'Corlu':'Çorlu','Edirne':'Edirne','Ipsala':'İpsala','Kirklareli':'Kırklareli',
            'Luleburgaz':'Lüleburgaz','Sariyer':'Sarıyer','Tekirdag':'Tekirdağ','Uzunkopru':'Uzunköprü'}
si = pd.read_excel('/mnt/user-data/uploads/Paper 3/data/istasyon bilgileri.xlsx')
si.columns = ['name','lat_dms','lon_dms','elev']
def dms2dd(s):
    d,m,sec,h = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", s.strip()).groups()
    return (float(d)+float(m)/60+float(sec)/3600)*(-1 if h in ('S','W') else 1)
si['lat'] = si.lat_dms.map(dms2dd); si['lon'] = si.lon_dms.map(dms2dd)
si['name'] = si['name'].map(NAME_MAP)

# Trakya sınırı
sf = shapefile.Reader(SHP)
polys, segs = [], []
for shp in sf.shapes():
    pts = np.array(shp.points)
    parts = list(shp.parts) + [len(pts)]
    for a, b in zip(parts[:-1], parts[1:]):
        ring = pts[a:b]
        polys.append(ring); segs.append(ring)

fig, ax = plt.subplots(figsize=(10.8, 7.6))

# kara dolgusu + sınır
ax.add_collection(PolyCollection(polys, facecolors='#F0F4E8', edgecolors='none', zorder=1))
ax.add_collection(LineCollection(segs, colors='#5B7A3A', linewidths=1.1, zorder=2))
ax.set_facecolor('#EAF2F8')  # deniz

# istasyonlar
sc = ax.scatter(si.lon, si.lat, c='#1F4E9C', s=150, marker='o', edgecolor='white',
                linewidth=1.4, zorder=6)
LABEL_OFF = {  # (dx, dy) points, va
    'Edirne': (0, 12, 'bottom'), 'Kırklareli': (0, 12, 'bottom'), 'Uzunköprü': (0, -13, 'top'),
    'Lüleburgaz': (0, 12, 'bottom'), 'Çorlu': (0, 12, 'bottom'), 'Tekirdağ': (-6, -13, 'top'),
    'İpsala': (-6, -13, 'top'), 'Sarıyer': (-24, 12, 'bottom'),
}
for _, r_ in si.iterrows():
    dx, dy, va = LABEL_OFF[r_['name']]
    txt = f"{r_['name']}\n({int(r_.elev)} m)"
    ax.annotate(txt, (r_.lon, r_.lat), xytext=(dx, dy), textcoords='offset points',
                ha='center', va=va, fontsize=9.5, fontweight='bold', color='#1F2937',
                path_effects=[pe.withStroke(linewidth=2.6, foreground='white')], zorder=7,
                linespacing=1.0)

# deniz / ülke etiketleri
ax.text(28.35, 41.86, 'BLACK SEA', fontsize=10, color='#3D6E9E', style='italic', ha='center')
ax.text(27.32, 40.66, 'SEA OF MARMARA', fontsize=10, color='#3D6E9E', style='italic', ha='center')
ax.text(25.95, 40.42, 'AEGEAN\nSEA', fontsize=10, color='#3D6E9E', style='italic', ha='center', linespacing=1.1)
ax.text(25.68, 41.35, 'GREECE', fontsize=9, color='#8A8F98', ha='center', rotation=90)
ax.text(27.20, 42.16, 'BULGARIA', fontsize=9, color='#8A8F98', ha='center')
ax.text(26.55, 40.62, 'Gallipoli\nPen.', fontsize=8, color='#6B7280', ha='center', style='italic', linespacing=1.0)

# Ergene havzası çekirdeği notu yok (nehir verisi yok) — sade tutuldu

# eksenler
ax.set_xlim(25.45, 29.35); ax.set_ylim(40.15, 42.30)
ax.set_aspect(1/np.cos(np.radians(41.2)))
ax.set_xlabel('Longitude (°E)', fontsize=11); ax.set_ylabel('Latitude (°N)', fontsize=11)
ax.tick_params(labelsize=10)
for s_ in ['top','right']: ax.spines[s_].set_visible(False)

# ölçek çubuğu (41.2°N'de 1° boylam ≈ 83.8 km) — 50 km
km_per_deg = 111.32*np.cos(np.radians(41.2))
L = 50/km_per_deg
x0, y0 = 25.62, 40.24
ax.plot([x0, x0+L], [y0, y0], color='#1F2937', lw=2.5, solid_capstyle='butt')
ax.plot([x0, x0], [y0-0.012, y0+0.012], color='#1F2937', lw=1.6)
ax.plot([x0+L, x0+L], [y0-0.012, y0+0.012], color='#1F2937', lw=1.6)
ax.text(x0+L/2, y0+0.03, '50 km', ha='center', fontsize=9, color='#1F2937')

# kuzey oku
ax.annotate('N', xy=(29.18, 42.02), xytext=(29.18, 41.80), ha='center', fontsize=11,
            fontweight='bold', color='#1F2937',
            arrowprops=dict(arrowstyle='-|>', color='#1F2937', lw=1.8))

# ---- inset: Türkiye konumu ----
axi = fig.add_axes([0.652, 0.150, 0.30, 0.235])
g = json.load(open(W+'ne50_countries.geojson'))
def draw_country(name, fc, ec, lw, z):
    for f in g['features']:
        if f['properties'].get('NAME') == name:
            geom = f['geometry']
            rings = geom['coordinates'] if geom['type']=='Polygon' else [c for p in geom['coordinates'] for c in [p]]
            if geom['type']=='Polygon':
                pls = [np.array(r) for r in geom['coordinates']]
            else:
                pls = [np.array(r) for poly in geom['coordinates'] for r in poly]
            axi.add_collection(PolyCollection([p for p in pls if p.ndim==2], facecolors=fc, edgecolors=ec, linewidths=lw, zorder=z))
for f in g['features']:
    geom = f['geometry']
    if geom['type']=='Polygon': pls = [np.array(r) for r in geom['coordinates']]
    else: pls = [np.array(r) for poly in geom['coordinates'] for r in poly]
    axi.add_collection(PolyCollection([p for p in pls if p.ndim==2], facecolors='#F2F2F0', edgecolors='#C9CCD1', linewidths=0.4, zorder=1))
draw_country('Turkey', '#DCE7CE', '#5B7A3A', 0.8, 2)
axi.add_patch(Rectangle((25.45, 40.15), 29.35-25.45, 42.30-40.15, fill=False, edgecolor='#C0392B', lw=1.6, zorder=5))
axi.text(35.2, 38.6, 'TÜRKİYE', fontsize=8.5, ha='center', color='#3A5222', fontweight='bold')
axi.text(23.5, 44.4, 'Study area', fontsize=7.5, ha='center', color='#C0392B')
axi.annotate('', xy=(26.8, 41.4), xytext=(24.4, 43.9), arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))
axi.set_xlim(19.5, 45.5); axi.set_ylim(33.6, 46.5)
axi.set_aspect(1/np.cos(np.radians(39)))
axi.set_xticks([]); axi.set_yticks([])
for s_ in axi.spines.values(): s_.set_edgecolor('#8A8F98')
axi.set_facecolor('#EAF2F8')

plt.tight_layout()
plt.savefig(W+'figs_EN/Fig1_study_area.png', dpi=300, bbox_inches='tight')
print('OK Fig1')
