# Adım 3: Run teorisi olay katalogları + kritik kuraklık şiddet serileri (D-sınıfları) + P0
# Tanımlar Kıvrak (2024) Bölüm 5 = Çavuş & Aksoy (2019, 2020) çerçevesi:
#  - kurak dönem: indeks < 0 ardışık aylar; D-aylık "dönem-içi" kuraklıklar pencerelerle
#  - yılın D-aylık kritik kuraklığı: o yıl biten D-pencereleri içinde maks kümülatif şiddet
#  - kuraklık olmayan yıl: S_D = 0 (toplam olasılık için P0)
import pandas as pd, numpy as np

W = "/home/claude/work/"
sp = pd.read_csv(W + "trakya_spi_spei_v2.csv")
STS = sorted(sp.name.unique())
IDX = ['SPI12', 'SPEI12']
YR0, YR1 = 1966, 2024          # 1965 başlangıç NaN'ları nedeniyle tam yıllar
NYRS = YR1 - YR0 + 1
DMAX = 12

events = []
crit = {}   # (name, idx, year, D) -> S
for st in STS:
    g = sp[sp.name==st].sort_values(['year','month']).reset_index(drop=True)
    for ix in IDX:
        v = g[ix].values
        dry = np.isfinite(v) & (v < 0)
        i = 0
        while i < len(v):
            if dry[i]:
                j = i
                while j+1 < len(v) and dry[j+1]: j += 1
                L = j - i + 1
                seg = -v[i:j+1]                       # |SPI| değerleri
                S = float(seg.sum())
                events.append({'name':st,'index':ix,
                               'bas_yil':int(g.year[i]),'bas_ay':int(g.month[i]),
                               'son_yil':int(g.year[j]),'son_ay':int(g.month[j]),
                               'L_ay':L,'S':round(S,3),'I':round(S/L,3),
                               'tepe':round(float(seg.max()),3)})
                # D-sınıfı pencereler (dönem-içi kuraklıklar)
                for D in range(1, min(L, DMAX)+1):
                    cs = np.concatenate([[0.0], np.cumsum(seg)])
                    for a in range(0, L-D+1):
                        sD = float(cs[a+D] - cs[a])
                        yr_end = int(g.year[i+a+D-1])    # pencerenin bittiği yıl
                        if yr_end < YR0: continue
                        key = (st, ix, yr_end, D)
                        if sD > crit.get(key, 0.0): crit[key] = sD
                i = j + 1
            else:
                i += 1

ev = pd.DataFrame(events)
ev.to_csv(W + "olay_katalogu.csv", index=False)

# Kritik şiddet serileri (tam yıl x D ızgarası; olmayan = 0)
rows = []
for st in STS:
    for ix in IDX:
        for yr in range(YR0, YR1+1):
            for D in range(1, DMAX+1):
                rows.append({'name':st,'index':ix,'year':yr,'D':D,
                             'S':round(crit.get((st,ix,yr,D), 0.0), 3)})
cs = pd.DataFrame(rows)
cs.to_csv(W + "kritik_siddet_serileri.csv", index=False)

# Özet: olay istatistikleri (T3 taslağı)
print("OLAY KATALOĞU ÖZETİ (eşik 0, tam seri):")
summ = ev.groupby(['name','index']).agg(n=('S','size'), ortS=('S','mean'), maksS=('S','max'),
                                        ortL=('L_ay','mean'), maksL=('L_ay','max')).round(2)
print(summ.to_string())

# P0 tablosu (D = 1, 3, 6, 9, 12)
print(f"\nP0 (sıfır-şiddet olasılığı, {NYRS} yıl):")
p0rows = []
for st in STS:
    for ix in IDX:
        sub = cs[(cs.name==st)&(cs['index']==ix)]
        r = {'name':st,'index':ix}
        for D in [1,3,6,9,12]:
            z = (sub[sub.D==D].S == 0).sum()
            r[f'D{D}'] = round(z/NYRS, 3)
        p0rows.append(r)
p0 = pd.DataFrame(p0rows)
print(p0.to_string(index=False))
p0.to_csv(W + "P0_tablosu.csv", index=False)

# En şiddetli 5 olay (SPEI12, tüm bölge)
print("\nEn şiddetli 5 olay (SPEI12):")
top = ev[ev['index']=='SPEI12'].nlargest(5, 'S')
print(top[['name','bas_yil','bas_ay','son_yil','son_ay','L_ay','S','tepe']].to_string(index=False))
print("\n2019-2024 olaylarında en uzun süreler (SPEI12):")
son = ev[(ev['index']=='SPEI12') & (ev.son_yil>=2019)].nlargest(5,'L_ay')
print(son[['name','bas_yil','bas_ay','son_yil','son_ay','L_ay','S']].to_string(index=False))
