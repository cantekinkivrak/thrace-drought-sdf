# Sıcaklık dolgu v2 — komşu-anomali (klimatoloji + r²-ağırlıklı komşu anomalisi) yöntemi
import pandas as pd, numpy as np, json

D3 = "/mnt/user-data/uploads/Paper 3/data/"
TPPET = ("/mnt/user-data/uploads/paper 2/files (60)/thrace-drought-scale-selection/"
         "thrace-drought-scale-selection/data/derived/trakya_TP_PET_1965_2024.csv")
MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
FILES = {"Çorlu":"corlu_temp.xlsx","Edirne":"edirne_temp.xlsx","İpsala":"ipsala_temp.xlsx",
         "Kırklareli":"kirklareli_temp.xlsx","Lüleburgaz":"luleburgaz_temp.xlsx",
         "Sarıyer":"sariyer_temp.xlsx","Tekirdağ":"tekirdag_temp.xlsx","Uzunköprü":"uzunkopru_temp.xlsx"}

# ---- 1) Uzun format yükle ----
rows = []
for st, f in FILES.items():
    x = pd.read_excel(D3 + f)
    l = x.melt(id_vars='year', value_vars=MON, var_name='mn', value_name='temp')
    l['month'] = l['mn'].map({m: i+1 for i, m in enumerate(MON)})
    l['name'] = st
    rows.append(l[['name','year','month','temp']])
df = pd.concat(rows, ignore_index=True)
df['temp'] = pd.to_numeric(df['temp'], errors='coerce')

# Uzunköprü 1965 Oca-Şub: hatalı kaynak değerleri → NaN (yeniden tahmin edilecek)
bad_mask = (df.name=='Uzunköprü') & (df.year==1965) & (df.month.isin([1,2]))
bad_orig = df.loc[bad_mask, ['name','year','month','temp']].copy()
df.loc[bad_mask, 'temp'] = np.nan

targets = df[df.temp.isna()][['name','year','month']].copy()
print(f"Doldurulacak ay sayısı: {len(targets)}")

# ---- 2) Klimatoloji ve anomaliler ----
clim = df.groupby(['name','month'])['temp'].mean().rename('clim')
df = df.merge(clim, on=['name','month'])
df['anom'] = df['temp'] - df['clim']

# Anomali korelasyon matrisi (istasyonlar arası, ortak aylar)
wide = df.pivot_table(index=['year','month'], columns='name', values='anom')
corr = wide.corr()
print("\nAnomali korelasyon matrisi (min-maks):", round(corr.min().min(),3), "-",
      round(corr.where(~np.eye(len(corr),dtype=bool)).max().max(),3))

# ---- 3) Tahmin: r²-ağırlıklı komşu anomali ort. + klimatoloji ----
stations = list(FILES.keys())
pred = pd.Series(index=wide.index, dtype=float)
preds = {}
for st in stations:
    others = [o for o in stations if o != st]
    W = (corr.loc[st, others]**2)
    A = wide[others]
    num = (A * W).sum(axis=1, skipna=True)
    den = A.notna().mul(W).sum(axis=1)
    preds[st] = num / den
pred_anom = pd.DataFrame(preds)

# ---- 4) Çapraz doğrulama (gözlemli aylarda rezidüel) ----
res = (wide - pred_anom)
flat = res.stack().rename('resid').reset_index()
rmse_all = np.sqrt((flat.resid**2).mean()); mae_all = flat.resid.abs().mean()
print(f"\nDoğrulama (tüm gözlemli aylar, n={len(flat)}): RMSE={rmse_all:.3f}°C, MAE={mae_all:.3f}°C")
per_st = flat.groupby('name').resid.agg(RMSE=lambda r: np.sqrt((r**2).mean()), MAE=lambda r: r.abs().mean(), n='size')
print(per_st.round(3).to_string())

# ---- 5) Aykırı değer taraması (|rezidüel| en büyükler) ----
flat['abs'] = flat.resid.abs()
out_scan = flat.sort_values('abs', ascending=False).head(12)
print("\nEn büyük 12 rezidüel (şüpheli değer taraması):")
for _, r in out_scan.iterrows():
    obs = df[(df.name==r['name'])&(df.year==r.year)&(df.month==r.month)].temp.iloc[0]
    print(f"  {r['name']:<12} {int(r.year)}-{int(r.month):02d}: gözlem={obs:.1f}°C, rezidüel={r.resid:+.2f}°C")

# ---- 6) Dolgu değerleri ----
clim_l = clim.reset_index()
fills = []
for _, t in targets.iterrows():
    pa = pred_anom.loc[(t.year, t.month), t['name']]
    cl = clim_l[(clim_l.name==t['name'])&(clim_l.month==t.month)].clim.iloc[0]
    nb = df[(df.year==t.year)&(df.month==t.month)&(df.name!=t['name'])].temp.mean()
    fills.append({'name':t['name'],'year':int(t.year),'month':int(t.month),
                  'yeni':round(cl+pa,2),'clim':round(cl,2),'komsu_ort':round(nb,2)})
fills = pd.DataFrame(fills)

# Eski değerler (TP_PET'teki PCHIP dolgular / ham hatalı değerler)
tp = pd.read_csv(TPPET)
fills = fills.merge(tp[['name','year','month','temp']].rename(columns={'temp':'eski'}), on=['name','year','month'], how='left')
fills['fark'] = (fills.eski - fills.yeni).round(2)
fills['tur'] = np.where((fills.name=='Uzunköprü')&(fills.year==1965), 'düzeltme (hatalı kaynak)', 'dolgu (eksik ay)')
print("\nESKİ vs YENİ dolgular:")
print(fills[['name','year','month','eski','yeni','komsu_ort','fark','tur']].to_string(index=False))

# ---- 7) Final seri + kaydet ----
df['temp_v2'] = df['temp']
for _, f in fills.iterrows():
    df.loc[(df.name==f['name'])&(df.year==f.year)&(df.month==f.month), 'temp_v2'] = f.yeni
df['kaynak'] = 'gözlem'
for _, f in fills.iterrows():
    df.loc[(df.name==f['name'])&(df.year==f.year)&(df.month==f.month), 'kaynak'] = f.tur
assert df.temp_v2.isna().sum() == 0, "Hâlâ eksik var!"

master = df[['name','year','month','temp_v2','kaynak']].rename(columns={'temp_v2':'temp'}).sort_values(['name','year','month'])
master.to_csv('/home/claude/work/trakya_temp_filled_v2.csv', index=False, float_format='%.2f')
fills.to_csv('/home/claude/work/fills_karsilastirma.csv', index=False)
per_st.round(3).to_csv('/home/claude/work/validasyon.csv')
out_scan[['name','year','month','resid']].to_csv('/home/claude/work/outlier_scan.csv', index=False)
corr.round(3).to_csv('/home/claude/work/korelasyon_matrisi.csv')
print("\nKaydedildi: trakya_temp_filled_v2.csv (", len(master), "satır)")

# İpsala 1976 yıllık ortalama etkisi
for st, yr in [('İpsala',1976),('İpsala',1978),('Uzunköprü',1965)]:
    old = tp[(tp.name==st)&(tp.year==yr)].temp.mean()
    new = master[(master.name==st)&(master.year==yr)].temp.mean()
    print(f"{st} {yr} yıllık ort: eski={old:.2f}°C → yeni={new:.2f}°C")
