# CMIP6 çıkarım: Trakya kutusu → istasyon-en yakın hücre aylık serileri (3 model × deneyler × pr/tas)
import xarray as xr, pandas as pd, numpy as np, re, glob, os, warnings
warnings.filterwarnings('ignore')

UP = "/mnt/user-data/uploads/Paper 3/cmip6_raw/"
WK = "/home/claude/work/cmip6/"
D3 = "/mnt/user-data/uploads/Paper 3/data/"
OUT = "/home/claude/work/"
LAT0, LAT1, LON0, LON1 = 39.0, 43.5, 24.5, 31.0

# İstasyon koordinatları
NAME_MAP = {'Corlu':'Çorlu','Edirne':'Edirne','Ipsala':'İpsala','Kirklareli':'Kırklareli',
            'Luleburgaz':'Lüleburgaz','Sariyer':'Sarıyer','Tekirdag':'Tekirdağ','Uzunkopru':'Uzunköprü'}
si = pd.read_excel(D3 + "istasyon bilgileri.xlsx"); si.columns = ['name','lat_dms','lon_dms','elev']
def dms2dd(s):
    d,m,sec,h = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", s.strip()).groups()
    return (float(d)+float(m)/60+float(sec)/3600)*(-1 if h in ('S','W') else 1)
si['lat'] = si.lat_dms.map(dms2dd); si['lon'] = si.lon_dms.map(dms2dd)
si['name'] = si['name'].map(NAME_MAP)

FILES = []
for f in sorted(glob.glob(UP + "*.nc")) + sorted(glob.glob(WK + "*.nc")):
    b = os.path.basename(f)
    m = re.match(r"(pr|tas)_Amon_([A-Za-z0-9-]+)_(historical|ssp245|ssp585)_", b)
    if m: FILES.append((m.group(1), m.group(2), m.group(3), f))
print(f"{len(FILES)} dosya işlenecek")

rows = []
for var, model, exp, path in FILES:
    ds = xr.open_dataset(path, use_cftime=True)
    da = ds[var].sel(lat=slice(LAT0, LAT1), lon=slice(LON0, LON1)).load()
    ds.close()
    yrs = np.array([t.year for t in da.time.values]); mos = np.array([t.month for t in da.time.values])
    keep = (yrs >= 1965) & (yrs <= 2014) if exp == 'historical' else (yrs >= 2015)
    da = da.isel(time=keep); yrs = yrs[keep]; mos = mos[keep]
    dim = np.array([[31,28,31,30,31,30,31,31,30,31,30,31][m-1] for m in mos])
    # istasyon → en yakın hücre
    for _, st in si.iterrows():
        cell = da.sel(lat=st['lat'], lon=st['lon'], method='nearest')
        v = cell.values.astype(float)
        if var == 'pr':
            v = v * 86400.0 * dim          # kg m-2 s-1 → mm/ay
        else:
            v = v - 273.15                 # K → °C
        rows.append(pd.DataFrame({'model':model,'exp':exp,'var':var,'name':st['name'],
                                  'year':yrs,'month':mos,'value':np.round(v, 3),
                                  'cell_lat':round(float(cell.lat),3),'cell_lon':round(float(cell.lon),3)}))
    print(f"  {model:<12} {exp:<10} {var}  ({keep.sum()} ay) hücre örneği: lat={float(cell.lat):.2f} lon={float(cell.lon):.2f}")

df = pd.concat(rows, ignore_index=True)
wide = df.pivot_table(index=['model','exp','name','year','month'], columns='var', values='value').reset_index()
wide = wide.rename(columns={'pr':'pr_mm','tas':'tas_c'})
wide.to_csv(OUT + "cmip6_trakya_istasyon_aylik.csv", index=False)
print(f"\nMaster: {len(wide)} satır → cmip6_trakya_istasyon_aylik.csv")
print("\nKapsam (model × deney, ay sayısı / istasyon):")
print(wide.groupby(['model','exp']).size().div(8).astype(int).to_string())

# Ham sapma kontrolü (1965-2014): model vs gözlem yıllık ortalamalar
obs = pd.read_csv(OUT + "trakya_TP_PET_v2.csv")
obs_ann = obs.groupby('name').agg(P_obs=('precip', lambda x: x.sum()/60), T_obs=('temp','mean'))
h = wide[wide.exp=='historical']
mod_ann = h.groupby(['model','name']).agg(P_mod=('pr_mm', lambda x: x.sum()/50), T_mod=('tas_c','mean')).reset_index()
mod_ann = mod_ann.merge(obs_ann, on='name')
mod_ann['dP_%'] = (100*(mod_ann.P_mod-mod_ann.P_obs)/mod_ann.P_obs).round(0)
mod_ann['dT_°C'] = (mod_ann.T_mod-mod_ann.T_obs).round(1)
print("\nHam sapmalar (QDM öncesi, bölge ort):")
print(mod_ann.groupby('model')[['dP_%','dT_°C']].mean().round(1).to_string())
