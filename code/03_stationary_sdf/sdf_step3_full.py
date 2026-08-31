# Tüm ölçekler (k=1,3,6,9,12) × SPI/SPEI: olay katalogları + kritik şiddet serileri + P0
import pandas as pd, numpy as np

W = "/home/claude/work/"
sp = pd.read_csv(W + "trakya_spi_spei_v2.csv")
STS = sorted(sp.name.unique())
KS = [1, 3, 6, 9, 12]
COLS = [(f'{fam}{k}', fam, k) for fam in ['SPI','SPEI'] for k in KS]
YR0, YR1 = 1966, 2024
NYRS = YR1 - YR0 + 1
DMAX = 12

events, crit = [], {}
for st in STS:
    g = sp[sp.name==st].sort_values(['year','month']).reset_index(drop=True)
    for col, fam, k in COLS:
        v = g[col].values
        dry = np.isfinite(v) & (v < 0)
        i = 0
        while i < len(v):
            if dry[i]:
                j = i
                while j+1 < len(v) and dry[j+1]: j += 1
                L = j - i + 1
                seg = -v[i:j+1]
                S = float(seg.sum())
                events.append({'name':st,'indeks':fam,'k':k,
                               'bas_yil':int(g.year[i]),'bas_ay':int(g.month[i]),
                               'son_yil':int(g.year[j]),'son_ay':int(g.month[j]),
                               'L_ay':L,'S':round(S,3),'I':round(S/L,3),'tepe':round(float(seg.max()),3)})
                csum = np.concatenate([[0.0], np.cumsum(seg)])
                for D in range(1, min(L, DMAX)+1):
                    for a in range(0, L-D+1):
                        sD = float(csum[a+D] - csum[a])
                        yr_end = int(g.year[i+a+D-1])
                        if yr_end < YR0: continue
                        key = (st, fam, k, yr_end, D)
                        if sD > crit.get(key, 0.0): crit[key] = sD
                i = j + 1
            else:
                i += 1

ev = pd.DataFrame(events)
ev.to_csv(W + "olay_katalogu_tum.csv", index=False)
print(f"Olay kataloğu: {len(ev)} olay (tüm ölçekler)")
print(ev.groupby(['indeks','k']).size().unstack().to_string())

rows = []
for st in STS:
    for col, fam, k in COLS:
        for yr in range(YR0, YR1+1):
            for D in range(1, DMAX+1):
                rows.append({'name':st,'indeks':fam,'k':k,'year':yr,'D':D,
                             'S':round(crit.get((st,fam,k,yr,D), 0.0),3)})
cs = pd.DataFrame(rows)
cs.to_csv(W + "kritik_siddet_serileri_tum.csv", index=False)
print(f"\nKritik şiddet serileri: {len(cs)} satır")

p0rows = []
for st in STS:
    for col, fam, k in COLS:
        sub = cs[(cs.name==st)&(cs.indeks==fam)&(cs.k==k)]
        r = {'name':st,'indeks':fam,'k':k}
        for D in [1,3,6,9,12]:
            r[f'P0_D{D}'] = round((sub[sub.D==D].S==0).mean(), 3)
        p0rows.append(r)
p0 = pd.DataFrame(p0rows)
p0.to_csv(W + "P0_tablosu_tum.csv", index=False)
print("\nP0 bölge ortalamaları (D=6):")
print(p0.groupby(['indeks','k'])['P0_D6'].mean().round(3).unstack().to_string())
