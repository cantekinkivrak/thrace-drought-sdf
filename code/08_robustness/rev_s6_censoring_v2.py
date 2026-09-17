# S6 (v2): boundary-event censoring and record-event exclusion in the copula analysis (Table S8).
# Reuses the copula/marginal functions of rev_s3_copula.py and adds, beside the "terminal event
# excluded" variant of the original run, a genuine "record event excluded" variant and "both".
# Run from the repository root.
import numpy as np, pandas as pd
src = open('code/08_robustness/rev_s3_copula.py', encoding='utf-8').read().split('# ---------- (a)')[0]
src = src.replace('W = "/home/claude/work/"', 'W = "data/derived/"')
exec(src)
recs = {('Tekirdağ','SPEI'):(90.016,72), ('Tekirdağ','SPI'):(70.735,55), ('İpsala','SPEI'):(92.731,72), ('İpsala','SPI'):(82.565,69)}
rows = []
for (st, fam), (s_star, d_star) in recs.items():
    g = ev[(ev.name==st)&(ev.indeks==fam)]
    S = g.S.values.astype(float); Dm = g.L_ay.values.astype(float)
    term = (g.son_yil>=2024).values; rec = np.isclose(S, s_star); assert rec.sum()==1
    lab = lambda m: ', '.join(f"{r.bas_yil}/{r.bas_ay:02d}–{r.son_yil}/{r.son_ay:02d}" for _, r in g[m].iterrows())
    for et, mask in [('full sample', np.ones(len(g), bool)), ('terminal event excluded', ~term),
                     ('record event excluded', ~rec), ('both excluded', ~(term|rec))]:
        tau, best = fit_copula(S[mask], Dm[mask]); T = t_and(S[mask], Dm[mask], s_star, d_star, best)
        rows.append(dict(st=st, idx=fam, variant=et, excluded=lab(~mask) if (~mask).any() else '—',
                         n=int(mask.sum()), tau=round(tau,3), cop=best[0], TAND=round(T,1)))
df = pd.DataFrame(rows); print(df.to_string(index=False))
df.to_csv('results/robustness/rev_copula_censoring_v2.csv', index=False)
