# Adım 9: Bileşik kuraklık risk endeksi (Pandya & Gontia genişletmesi) — SPEI tabanlı
# C1 tarihsel şiddet (T50, D=6/9/12 ort) · C2 şiddetlenme (MK z ort) · C3 sıklaşma (−P0 eğimi) · C4 gelecek değişim (ΔT50 %)
import pandas as pd, numpy as np

W = "/home/claude/work/"
sdf = pd.read_csv(W + "sdf_fits_tum.csv")
mk  = pd.read_csv(W + "mk_trend_sonuclari.csv")
ns  = pd.read_csv(W + "nsgev_sonuclari.csv")
dg  = pd.read_csv(W + "gelecek_sdf_degisim.csv")

STS = sorted(sdf.name.unique())

# C1: tarihsel SPEI T50 (k=12, D∈{6,9,12} mevcut olanların ortalaması)
c1 = (sdf[(sdf.indeks=='SPEI')&(sdf.k==12)&(sdf.D.isin([6,9,12]))]
      .groupby('name')['T50'].mean().rename('C1_tarihsel_T50'))

# C2: MK z ortalaması (SPEI, k=12, D∈{3,6,9,12})
c2 = (mk[(mk.indeks=='SPEI')&(mk.k==12)&(mk.D.isin([3,6,9,12]))]
      .groupby('name')['MK_z'].mean().rename('C2_MK_z'))

# C3: kuraklık yılı sıklaşması = −P0 eğimi ortalaması (SPEI, D∈{6,12})
c3 = (ns[(ns.indeks=='SPEI')&(ns.D.isin([6,12]))]
      .groupby('name')['P0_egim'].mean().mul(-1).rename('C3_siklasma'))

# C4: gelecek ΔT50% medyanı (SPEI; ssp245×2030-59 ve ssp585×2070-99; D∈{6,12})
sel = dg[(dg.indeks=='SPEI') & (dg.D.isin([6,12])) &
         (((dg.ssp=='ssp245')&(dg.pencere=='2030-2059')) | ((dg.ssp=='ssp585')&(dg.pencere=='2070-2099')))]
c4 = sel.groupby('name')['dT50_%'].median().rename('C4_gelecek_dT50')

tab = pd.concat([c1, c2, c3, c4], axis=1).reindex(STS)
print("Ham bileşenler:")
print(tab.round(2).to_string())

# Min-max normalizasyon (0 = en düşük risk, 1 = en yüksek)
norm = (tab - tab.min()) / (tab.max() - tab.min())
norm.columns = [c + '_n' for c in norm.columns]
tab = tab.join(norm)
tab['BILESIK'] = norm.mean(axis=1).round(3)
tab['SIRA'] = tab.BILESIK.rank(ascending=False).astype(int)
tab = tab.sort_values('BILESIK', ascending=False)
tab.round(3).to_csv(W + "risk_endeksi.csv")

print("\nBİLEŞİK RİSK SIRALAMASI (SPEI tabanlı):")
print(tab[['C1_tarihsel_T50','C2_MK_z','C3_siklasma','C4_gelecek_dT50','BILESIK','SIRA']].round(2).to_string())

# Ağırlık duyarlılığı: her bileşeni sırayla 2x ağırlıkla dene → sıralama ne kadar oynuyor?
print("\nAğırlık duyarlılığı (bileşen 2x iken ilk 3):")
for i, c in enumerate(norm.columns):
    wgt = np.ones(4); wgt[i] = 2
    comp = (norm * wgt).sum(axis=1) / wgt.sum()
    top3 = comp.sort_values(ascending=False).head(3).index.tolist()
    print(f"  {c:<22}: {' > '.join(top3)}")
