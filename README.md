# thrace-drought-sdf

Analysis code and derived datasets for:

> Kıvrak, C., Şener, M. — *From static to dynamic drought hazard: nonstationary and copula-based severity–duration–frequency curves for Thrace, Türkiye, under observed and projected warming.* Submitted to **Stochastic Environmental Research and Risk Assessment**.

The study builds drought severity–duration–frequency (SDF) curves for the Thrace region of Türkiye (8 stations, 1965–2024) from SPI and SPEI at five accumulation scales, using the critical-drought / total-probability framework, and extends them in four directions: (i) nonstationary frequency analysis (time-varying GEV location + logistic drought-occurrence model), (ii) copula-based joint severity–duration return periods, (iii) future SDF curves from three bias-corrected CMIP6 models (SSP2-4.5 / SSP5-8.5), and (iv) a composite drought hazard-priority index for adaptation screening.

## Repository layout

```
code/
  01_data_preparation/   temperature gap filling (seasonality-aware neighbour-anomaly
                         method) and a-posteriori detection of PCHIP-filled precipitation
  02_indices/            Thornthwaite PET, SPI (Thom gamma, zero-mix) and SPEI
                         (log-logistic, plotting-position PWM) at k = 1, 3, 6, 9, 12 months
  03_stationary_sdf/     run theory, critical drought severity series (D = 1–12),
                         seven-candidate distribution fitting (AD selection),
                         total-probability SDF curves, nonparametric bootstrap CIs
  04_nonstationary/      Mann–Kendall screen, NS-GEV (mu(t) = b0 + b1*t'),
                         logistic occurrence model, effective return levels
  05_copula/             event catalogue, five one-parameter copulas
                         (tau inversion, CvM selection), T_AND / T_OR of record events
  06_cmip6/              nearest-cell extraction from CMIP6 Amon files and
                         quantile delta mapping (QDM) per station/model/scenario/month
  07_hazard_priority/    composite hazard-priority index (C1–C4, min–max
                         normalized, unweighted mean) and robustness (LOCO, Dirichlet)
  08_robustness/         the full supplementary robustness programme S1–S15
                         (modified MK + FDR, bootstrap GOF, copula censoring/power/
                         forced-family, Oudin PET, early calibration, future
                         uncertainty, QDM preservation, window-year, shape-fixed refits)
  09_figures/            all manuscript (Fig. 1–8) and supplementary (S1–S2) figures
data/
  derived/               SPI/SPEI series, critical severity series, event catalogues,
                         P0 tables, QC/homogeneity diagnostics
  cmip6_station_series/  station-cell CMIP6 series (raw and QDM-corrected) and
                         future SPI/SPEI-12
results/
  sdf/                   fitted SDF quantiles (all cells) and bootstrap CIs
  trends_nonstationary/  MK results and NS-GEV parameter tables
  copula/                copula fits and record-event joint return periods
  future/                future SDF fits and relative changes
  hazard_priority/       component and composite index values
  robustness/            all rev_*.csv outputs behind Supplement S1–S15 and
                         audit_counts.json (frozen canonical significance counts)
figures/                 300-dpi PNG versions of all figures
```

## Data availability and reproducibility

**Raw station observations are not included.** The monthly precipitation and temperature records of the eight stations (Edirne 17050, Kırklareli 17052, Çorlu 17054, Tekirdağ 17056, Sarıyer 17061, Uzunköprü 17608, Lüleburgaz 17631, İpsala 17632) belong to the Turkish State Meteorological Service (MGM) and cannot be redistributed by the authors; equivalent data can be requested from MGM (https://www.mgm.gov.tr). Raw CMIP6 model output (GFDL-ESM4, MRI-ESM2-0, ACCESS-CM2; historical, SSP2-4.5, SSP5-8.5; r1i1p1f1 Amon) is openly available from the Earth System Grid Federation (https://esgf.llnl.gov); the exact dataset versions are listed in Table 2 of the manuscript.

Everything downstream of the raw observations is provided: the standardized index series, every intermediate derived dataset, every result table in the manuscript and supplement, and the complete code chain that produced them. With the MGM data in place, the numbered `code/` folders reproduce the entire analysis in order. Scripts were written for a Linux analysis container and use absolute paths (`/home/claude/work/...`); adjust the path constants at the top of each script to your environment.

The columns of several derived files carry Turkish names kept for provenance; the most frequent are: `istasyon` = station, `indeks` = index (SPI/SPEI), `olcek` = accumulation scale k, `sure`/`D` = duration (months), `siddet` = severity, `yil` = year, `dagilim` = distribution, `donem` = period, `senaryo` = scenario.

## Requirements

Python ≥ 3.10 with `numpy`, `pandas`, `scipy`, `matplotlib`, `xarray`, `netcdf4`, `pyshp` (see `requirements.txt`). No compiled or proprietary dependencies.

## Citation

If you use this code or the derived datasets, please cite the paper (see `CITATION.cff`) and the archived release (Zenodo DOI on the repository page; all versions: https://doi.org/10.5281/zenodo.22201076).

## License

Code is released under the MIT License (see `LICENSE`). Derived datasets and figures are released under CC BY 4.0. The underlying raw meteorological observations remain the property of MGM.

## Changelog

- **v1.1.0 (2026-09-17)** — pre-submission corrections to the robustness programme (Stochastic Environmental Research and Risk Assessment submission):
  - `code/08_robustness/rev_s9_future_v2.py` → `results/robustness/rev_future_uncertainty_v2.csv`: year-resampling bootstrap of the future SDF changes recomputed with exactly the Table 10 statistic (AD-selected family per cell held fixed and refitted, model median, station change, regional median; 1,000 replicates). Supersedes `rev_future_uncertainty*.csv` from `rev_s6_future.py`, which used an L-moment/MLE GEV on the regional median series and is kept for provenance only.
  - `code/08_robustness/rev_s6_censoring_v2.py` → `results/robustness/rev_copula_censoring_v2.csv`: copula boundary-event censoring extended with a genuine record-event exclusion and a both-excluded variant, with the withheld event dates listed (Table S8). Supersedes `rev_copula_censoring.csv`.
  - `data/derived/validasyon.csv`: per-station validation sample sizes corrected to the observed station-months (5,744 in total; the 16 imputed/corrected cells were previously counted).
  - README and CITATION.cff: SPEI PWM description (plotting-position estimator, (i − 0.35)/n), target journal, repository URL.
- **v1.0.0 (2026-08-31)** — submission release.
