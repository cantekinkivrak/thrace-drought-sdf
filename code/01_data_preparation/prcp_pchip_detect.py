# Yağış serilerinde PCHIP dolgu tespiti: tekil + ardışık boşluk taraması (LOO yeniden-üretim testi)
import pandas as pd, numpy as np
from scipy.interpolate import PchipInterpolator

D3 = "/mnt/user-data/uploads/Paper 3/data/"
MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
FILES = {"Çorlu":"corlu_prcp_filled.csv","Edirne":"edirne_prcp_filled.csv","İpsala":"ipsala_prcp_filled.csv",
         "Kırklareli":"kirklareli_prcp_filled.csv","Lüleburgaz":"luleburgaz_prcp_filled.csv",
         "Sarıyer":"sariyer_prcp_filled.csv","Tekirdağ":"tekirdagp_filled.csv","Uzunköprü":"uzunkopru_prcp_filled.csv"}
FILES["Tekirdağ"] = "tekirdagp_filled.csv"
TOL = 0.06  # 1 ondalık yuvarlama payı

def load(st):
    df = pd.read_csv(D3 + FILES[st])
    l = df.melt(id_vars='year', value_vars=MON, var_name='mn', value_name='p')
    l['month'] = l['mn'].map({m:i+1 for i,m in enumerate(MON)})
    return l.sort_values(['year','month']).reset_index(drop=True)

results = []
for st in FILES:
    s = load(st); y = s.p.values.astype(float); n = len(y); idx = np.arange(n)
    hits = []  # (start, length, max_resid)
    for L in range(1, 9):                     # boşluk uzunluğu 1..8 ay
        for a in range(1, n - L - 1):         # uçlarda dolgu yapılamaz (interp için iki taraf gerekli)
            span = np.arange(a, a + L)
            mask = np.ones(n, bool); mask[span] = False
            pred = PchipInterpolator(idx[mask], y[mask], extrapolate=True)(span)
            resid = np.abs(y[span] - pred)
            if resid.max() < TOL:
                hits.append((a, L, resid.max()))
    # Maksimal aralıkları seç (uzun olan kısa olanı kapsıyorsa uzunu al; çakışanları birleştirme)
    hits.sort(key=lambda h: (-h[1], h[2]))
    taken = np.zeros(n, bool); final = []
    for a, L, r in hits:
        span = np.arange(a, a+L)
        if not taken[span].any():
            taken[span] = True; final.append((a, L, r))
    for a, L, r in sorted(final):
        yr, mo = int(s.year[a]), int(s.month[a])
        yre, moe = int(s.year[a+L-1]), int(s.month[a+L-1])
        vals = ", ".join(f"{v:.1f}" for v in y[a:a+L])
        results.append({'istasyon': st, 'bas': f"{yr}-{mo:02d}", 'son': f"{yre}-{moe:02d}",
                        'uzunluk': L, 'deger_mm': vals, 'maks_resid': round(float(r),3)})

res = pd.DataFrame(results)
if len(res):
    print(res.to_string(index=False))
else:
    print("Hiç dolgu tespit edilmedi.")
res.to_csv('/home/claude/work/prcp_pchip_tespit.csv', index=False)
print(f"\nToplam tespit: {len(res)} aralık, {res.uzunluk.sum() if len(res) else 0} ay")
