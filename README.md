# behavioral-drift-iot

Code and results for the paper

> A. A. Abdulhussein, M. A. Azzawi, M. F. Nasrudin.
> **Early Warning of Behavioral Drift in Edge-Deployed Deep Learning for IoT: A Lightweight Composite Drift Score Evaluated on Two Real Deployments.** Submitted to *Machine Learning* (Springer), 2026.

A compact 8-16-8-1 multilayer perceptron (289 parameters) is trained on an initial reference window, frozen, and then monitored on two real IoT deployments with three lightweight signals: the Jensen-Shannon divergence of the inputs (JSD), the prediction-spread deviation (SPD) and the calibration gap (CAL). Their equal-weight z-score average is the Composite Drift Score (DS). The pipeline tests whether these signals track and anticipate the RMSE of the frozen model.

## Reproduce everything

```bash
pip install -r requirements.txt
python run_all.py          # all tables -> results/
python make_figures.py     # Figs. 3-7  -> figures/
```

Both steps take well under a minute on a laptop. `results/summary.json` holds every scalar statistic quoted in the paper; the CSV files hold the tables.

## Data

| Dataset | File expected | Source |
|---|---|---|
| UCI Air Quality (De Vito et al. 2008) | `data/AirQualityUCI.csv` (included, CC BY 4.0) | https://archive.ics.uci.edu/dataset/360/air+quality |
| Intel Berkeley Research Lab (Madden 2004) | `data/intel_lab_data.txt` (not included, 150 MB) | http://db.csail.mit.edu/labdata/labdata.html |

For the Intel Lab data, download `data.txt.gz` from the source page, unzip it and save it as `data/intel_lab_data.txt`. If the file is missing, `run_all.py` runs the UCI analysis only. The derived hourly aggregates used in the paper are included as `results/intel/hourly_aggregates.csv`.

## What the pipeline does

1. **Load and clean.** UCI: drop rows with the `-200` missing-value sentinel in any used column (8,991 of 9,357 hourly rows remain). Intel Lab: keep motes 1-54 with complete readings in physically plausible ranges (5-50 °C, 0-105 % RH, 2.0-3.2 V), aggregate to hourly building-level features, and keep the leading run of days on which at least half of the 54 motes still report (24 days, 535 hourly records).
2. **Train and freeze.** Train on the first 4 weeks (UCI) or 3 days (Intel Lab) with Adam, lr 2e-3, 200 epochs, batch 128, He-normal initialization; seeds 0 (init) and 42 (shuffling).
3. **Monitor.** Per window (week or day): RMSE, MAE, JSD (20-bin histograms, add-one smoothing), SPD, CAL, DS.
4. **Analyse.** Descriptives, Cronbach's alpha, Bonferroni-corrected correlations, hierarchical regression with VIFs, epoch ANOVA, lead-time regressions for k = 1..4 with Newey-West HAC errors, detrending and first-difference checks, a persistence baseline with partial correlations, burn-in (causal) standardization, component ablation, reactive baselines (ADWIN, Page-Hinkley, DDM) and a ten-seed sweep.

## Repository layout

```
drift_pipeline.py   model, drift metrics, statistics, baselines
run_all.py          loaders and the full analysis for both datasets
make_figures.py     Figs. 3-7 of the paper
data/               UCI csv (+ your copy of the Intel Lab file)
results/uci/        window_metrics, descriptives, correlations, lead_time, ...
results/intel/      same files for the Intel Lab deployment
results/summary.json
figures/            PDF and PNG versions of Figs. 3-7
original_uci_run/   the authors' first UCI-only script and its outputs
```

The UCI branch of `drift_pipeline.py` is numerically identical to `original_uci_run/run_analysis_real.py`: the weekly metrics agree to 1e-15.

## Main results

| | UCI Air Quality | Intel Berkeley Lab |
|---|---|---|
| Windows | 52 weekly | 21 daily |
| RMSE first -> last window | 1.06 -> 5.15 µg/m³ | 1.15 -> 16.6 °C |
| DS(t) vs RMSE(t+1) | r = +0.50 (Newey-West p = .008) | r = +0.66 (p = .001) |
| ... after linear detrending | r = +0.40 (p = .004) | r = -0.37 (n.s.) |
| Label-free JSD(t) vs RMSE(t+1) | r = +0.51 | r = +0.84 |
| Persistence RMSE(t) vs RMSE(t+1), needs all labels | r = +0.64 | r = +0.96 |
| Dominant correlate of input shift | seasonal departure of T and AH from reference (r = +0.71, +0.75) | battery voltage (r = -0.88) |

## Citation

See `CITATION.cff`. Please also cite the two datasets.

## License

Code: MIT (see `LICENSE`). The UCI Air Quality data are distributed by the UCI Machine Learning Repository under CC BY 4.0.
