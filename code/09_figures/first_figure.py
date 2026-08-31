# İlk bakış figürü: SPI-12 vs SPEI-12 ısı haritaları + fark paneli (ısınma sinyali)
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = "/home/claude/work/"
out = pd.read_csv(W + "trakya_spi_spei_v2.csv")
STS = ['Edirne','Uzunköprü','İpsala','Kırklareli','Lüleburgaz','Çorlu','Tekirdağ','Sarıyer']  # batı→doğu yakın sıra
out['t'] = out.year + (out.month-0.5)/12

def grid(col):
    g = out.pivot_table(index='name', columns='t', values=col)
    return g.reindex(STS)

spi, spei = grid('SPI12'), grid('SPEI12')
diff = spei - spi
tt = spi.columns.values

fig, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True,
                         gridspec_kw={'height_ratios':[1,1,1]})
plt.rcParams['font.family'] = 'DejaVu Sans'

for ax, G, ttl, vlim, cmap in [
    (axes[0], spi,  'A — SPI-12 (yalnız yağış)', 2.5, 'RdBu'),
    (axes[1], spei, 'B — SPEI-12 (yağış − PET: sıcaklık etkisi dahil)', 2.5, 'RdBu'),
    (axes[2], diff, 'C — Fark: SPEI-12 − SPI-12  (kırmızı = ısınma kuraklığı şiddetlendiriyor)', 1.2, 'RdBu')]:
    tedges = np.append(tt - 1/24, tt[-1] + 1/24)
    im = ax.pcolormesh(tedges, np.arange(len(STS)+1), G.values,
                       cmap=cmap, vmin=-vlim, vmax=vlim, shading='flat')
    ax.set_yticks(np.arange(len(STS))+0.5); ax.set_yticklabels(STS, fontsize=8.5)
    ax.set_title(ttl, fontsize=10.5, loc='left')
    ax.invert_yaxis()
    cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=12)
    cb.ax.tick_params(labelsize=8)
    for s in ['top','right']: ax.spines[s].set_visible(False)

axes[2].set_xlabel('Yıl')
axes[0].set_xlim(1966, 2025)
plt.tight_layout()
plt.savefig(W + "spi12_spei12_ilk_bakis.png", dpi=200, bbox_inches='tight')
print("Figür kaydedildi.")

# Sayısal özet: SPEI12-SPI12 farkının on yıllık evrimi + kuraklık ayı sayıları
out['fark'] = out.SPEI12 - out.SPI12
out['dekad'] = (out.year//10)*10
dk = out.groupby('dekad')['fark'].mean().round(3)
print("\nSPEI12 − SPI12 dekadal ortalama (ısınma sinyali):")
print(dk.to_string())
for esik, ad in [(-1.0,'orta+ (≤ -1)'), (-1.5,'şiddetli+ (≤ -1.5)'), (-2.0,'ekstrem (≤ -2)')]:
    a = (out.SPI12 <= esik).sum(); b = (out.SPEI12 <= esik).sum()
    print(f"Kurak ay sayısı {ad}: SPI12={a}, SPEI12={b} ({(b-a)/max(a,1)*100:+.0f}%)")
son = out[out.year >= 2019]
print(f"\n2019-2024: SPI12 ort={son.SPI12.mean():+.2f}, SPEI12 ort={son.SPEI12.mean():+.2f}")
