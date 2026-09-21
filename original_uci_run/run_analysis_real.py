"""
Real-data empirical study: behavioral drift in an edge-deployed DL model.

Dataset: UCI Air Quality (De Vito et al. 2008).
  - 9358 hourly observations of 5 metal-oxide sensor responses
    (PT08.S1...PT08.S5) co-located with calibrated ground-truth
    concentrations (CO, NMHC, C6H6, NOx, NO2) and ambient temperature T,
    relative humidity RH, and absolute humidity AH.
  - 14 months of continuous operation (Mar 2004 - Apr 2005) at a
    polluted Italian city site.
  - Missing values encoded as -200.

Setup:
  - Target (DV for the regression network): C6H6(GT) (benzene, ground truth)
  - Features (8): PT08.S1..PT08.S5, T, RH, AH
  - Model: 2-layer MLP (8->16->8->1) in pure NumPy, MSE loss, Adam.
  - Train on the first 4 weeks (672 hours), then deploy with FROZEN
    weights and roll forward week by week.

Analyses (SPSS-equivalent, all implemented on NumPy/pandas):
  - Descriptive statistics (Table II)
  - Cronbach's alpha for three composite drift indices (Table III)
  - Pearson & Spearman correlations with Bonferroni correction (Table IV)
  - Hierarchical multiple regression, 3 blocks (Table V)
  - Repeated-measures ANOVA across 5 epoch bins (Table VI)
  - Lagged regression: Drift Score (week t) -> RMSE (week t+k)  (Table VII)

Outputs saved to /sessions/amazing-sharp-allen/mnt/outputs/practical/results/
"""

import os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path('/sessions/amazing-sharp-allen/mnt/outputs/practical')
DATA = BASE / 'data' / 'AirQualityUCI.csv'
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True, parents=True)

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. LOAD AND CLEAN REAL UCI AIR QUALITY DATA
# ---------------------------------------------------------------

raw = pd.read_csv(DATA, sep=';', decimal=',',
                  na_values=['-200', '-200.0', '-200,0'])
# Drop trailing empty unnamed columns + trailing all-NaN rows
raw = raw.dropna(axis=1, how='all')
raw = raw.dropna(subset=['Date']).reset_index(drop=True)

# Parse timestamp
raw['Timestamp'] = pd.to_datetime(
    raw['Date'] + ' ' + raw['Time'].str.replace('.', ':', regex=False),
    format='%d/%m/%Y %H:%M:%S')
raw = raw.sort_values('Timestamp').reset_index(drop=True)

# Hour index from start of deployment
t0 = raw['Timestamp'].iloc[0]
raw['hours'] = ((raw['Timestamp'] - t0).dt.total_seconds() / 3600.0).astype(int)
raw['week'] = raw['hours'] // (24 * 7)
raw['hour'] = raw['Timestamp'].dt.hour
raw['dow']  = raw['Timestamp'].dt.dayofweek

FEATS = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)',
         'PT08.S5(O3)', 'T', 'RH', 'AH']
TARGET = 'C6H6(GT)'

df = raw[['Timestamp', 'hours', 'week', 'hour', 'dow'] + FEATS + [TARGET]].copy()
n_before = len(df)
df = df.dropna(subset=FEATS + [TARGET]).reset_index(drop=True)
n_after = len(df)
print(f"Loaded {n_before} hourly rows; {n_after} retained after missing-value filtering "
      f"({n_before - n_after} dropped).")

# Convenience aliases used downstream (mirrors simulated-study column names)
df = df.rename(columns={
    'PT08.S1(CO)': 'S1', 'PT08.S2(NMHC)': 'S2', 'PT08.S3(NOx)': 'S3',
    'PT08.S4(NO2)': 'S4', 'PT08.S5(O3)': 'S5',
})

# Save cleaned stream
df.to_csv(RESULTS / 'uci_stream_clean.csv', index=False)

# ---------------------------------------------------------------
# 2. TRAIN COMPACT MLP ON FIRST 4 WEEKS (NumPy)
# ---------------------------------------------------------------

FEAT_COLS = ['S1', 'S2', 'S3', 'S4', 'S5', 'T', 'RH', 'AH']
X_all = df[FEAT_COLS].values.astype(np.float64)
y_all = df[TARGET].values.astype(np.float64)

TRAIN_WEEKS = 4
train_mask = (df['week'] < TRAIN_WEEKS).values

# Normalize using training-window stats
mu = X_all[train_mask].mean(axis=0)
sd = X_all[train_mask].std(axis=0) + 1e-9
Xn = (X_all - mu) / sd
ymu = y_all[train_mask].mean()
ysd = y_all[train_mask].std() + 1e-9
yn  = (y_all - ymu) / ysd

Xtr, ytr = Xn[train_mask], yn[train_mask]

def init_mlp(d_in=8, h1=16, h2=8, d_out=1, seed=0):
    r = np.random.default_rng(seed)
    return {
        'W1': r.normal(0, np.sqrt(2/d_in), (d_in, h1)), 'b1': np.zeros(h1),
        'W2': r.normal(0, np.sqrt(2/h1),   (h1, h2)),   'b2': np.zeros(h2),
        'W3': r.normal(0, np.sqrt(2/h2),   (h2, d_out)),'b3': np.zeros(d_out),
    }

def relu(x): return np.maximum(0, x)

def forward(p, X):
    z1 = X @ p['W1'] + p['b1']; a1 = relu(z1)
    z2 = a1 @ p['W2'] + p['b2']; a2 = relu(z2)
    z3 = a2 @ p['W3'] + p['b3']
    cache = (X, z1, a1, z2, a2, z3)
    return z3.squeeze(-1), cache

def backward(p, cache, y):
    X, z1, a1, z2, a2, z3 = cache
    N = X.shape[0]
    dz3 = (z3.squeeze(-1) - y).reshape(-1, 1) / N
    g = {}
    g['W3'] = a2.T @ dz3; g['b3'] = dz3.sum(0)
    da2 = dz3 @ p['W3'].T
    dz2 = da2 * (z2 > 0)
    g['W2'] = a1.T @ dz2; g['b2'] = dz2.sum(0)
    da1 = dz2 @ p['W2'].T
    dz1 = da1 * (z1 > 0)
    g['W1'] = X.T @ dz1; g['b1'] = dz1.sum(0)
    return g

def adam_step(p, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, t=1):
    for k in p:
        if k not in state:
            state[k] = {'m': np.zeros_like(p[k]), 'v': np.zeros_like(p[k])}
        s = state[k]
        s['m'] = b1 * s['m'] + (1 - b1) * g[k]
        s['v'] = b2 * s['v'] + (1 - b2) * (g[k] ** 2)
        mh = s['m'] / (1 - b1 ** t)
        vh = s['v'] / (1 - b2 ** t)
        p[k] -= lr * mh / (np.sqrt(vh) + eps)

p = init_mlp(seed=0)
state = {}
EPOCHS, BATCH = 200, 128
for epoch in range(1, EPOCHS + 1):
    idx = rng.permutation(len(Xtr))
    losses = []
    for i in range(0, len(idx), BATCH):
        b = idx[i:i + BATCH]
        yp, cache = forward(p, Xtr[b])
        loss = 0.5 * np.mean((yp - ytr[b]) ** 2)
        g = backward(p, cache, ytr[b])
        adam_step(p, g, state, lr=2e-3, t=epoch)
        losses.append(loss)
    if epoch % 40 == 0:
        yp_all, _ = forward(p, Xtr)
        rmse_tr = np.sqrt(np.mean((yp_all - ytr) ** 2)) * ysd
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  train RMSE {rmse_tr:.4f}")

# ---------------------------------------------------------------
# 3. DEPLOY (rolling weekly evaluation with FROZEN weights)
# ---------------------------------------------------------------

yhat_all, _ = forward(p, Xn)
yhat_all = yhat_all * ysd + ymu
df['yhat'] = yhat_all
df['err']  = df[TARGET] - df['yhat']
df['abs_err'] = np.abs(df['err'])

# Reference window stats (first TRAIN_WEEKS)
ref_mask = (df['week'] < TRAIN_WEEKS).values
ref_X = Xn[ref_mask]

def hist_pdf(x, bins):
    h, _ = np.histogram(x, bins=bins, density=False)
    h = h + 1.0  # Laplace smoothing
    return h / h.sum()

def js_div(p_, q_):
    m = 0.5 * (p_ + q_)
    def kl(a, b): return np.sum(np.where(a > 0, a * np.log(a / b), 0.0))
    return 0.5 * kl(p_, m) + 0.5 * kl(q_, m)

BINS = 20
feat_bins = []
ref_hists = []
for j in range(Xn.shape[1]):
    lo, hi = np.min(Xn[:, j]), np.max(Xn[:, j])
    edges = np.linspace(lo, hi, BINS + 1)
    feat_bins.append(edges)
    ref_hists.append(hist_pdf(ref_X[:, j], edges))

ref_pred_var = df.loc[df['week'] < TRAIN_WEEKS, 'yhat'].std()

weeks = sorted(df['week'].unique())
weekly = []
for w in weeks:
    mask = (df['week'] == w).values
    cur_X = Xn[mask]
    err = df.loc[mask, 'err'].values
    abs_err = df.loc[mask, 'abs_err'].values
    if mask.sum() < 80:  # skip undersampled weeks
        continue
    js = np.mean([js_div(hist_pdf(cur_X[:, j], feat_bins[j]), ref_hists[j])
                  for j in range(cur_X.shape[1])])
    pred_var = abs(df.loc[mask, 'yhat'].std() - ref_pred_var)
    calib_gap = abs(df.loc[mask, TARGET].mean() - df.loc[mask, 'yhat'].mean())
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(abs_err)
    weekly.append({
        'week': int(w),
        't_weeks': int(w),
        'RMSE': rmse, 'MAE': mae,
        'JS': js, 'PredVar': pred_var, 'CalibGap': calib_gap,
        'T_mean':  df.loc[mask, 'T'].mean(),
        'RH_mean': df.loc[mask, 'RH'].mean(),
        'AH_mean': df.loc[mask, 'AH'].mean(),
        'S5_mean': df.loc[mask, 'S5'].mean(),
        'n_samples': int(mask.sum()),
    })

W = pd.DataFrame(weekly).reset_index(drop=True)
W = W[W['n_samples'] >= 80].reset_index(drop=True)

# Standardize drift components to z-scores for composite score
for c in ['JS', 'PredVar', 'CalibGap']:
    W[c + '_z'] = (W[c] - W[c].mean()) / W[c].std()
W['DriftScore'] = W[['JS_z', 'PredVar_z', 'CalibGap_z']].mean(axis=1)

# Skip the training window for analysis; re-index from 0
Wd = W[W['week'] >= TRAIN_WEEKS].reset_index(drop=True).copy()
Wd['t_weeks'] = Wd['week'] - TRAIN_WEEKS
Wd.to_csv(RESULTS / 'weekly_metrics.csv', index=False)

print(f"\nWeekly windows analyzed: {len(Wd)}")
print(Wd[['week', 'RMSE', 'JS', 'PredVar', 'CalibGap', 'DriftScore']].head())
print(Wd[['week', 'RMSE', 'JS', 'PredVar', 'CalibGap', 'DriftScore']].tail())

# ---------------------------------------------------------------
# 4. SPSS-EQUIVALENT ANALYSES
# ---------------------------------------------------------------

def describe(x):
    x = np.asarray(x, float)
    n = len(x); m = x.mean(); s = x.std(ddof=1); med = np.median(x)
    q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    sk = ((x - m) ** 3).mean() / (s ** 3 + 1e-12)
    kt = ((x - m) ** 4).mean() / (s ** 4 + 1e-12) - 3
    return dict(N=n, Mean=m, SD=s, Median=med, IQR=iqr, Skew=sk, Kurt=kt)

desc_vars = ['t_weeks', 'T_mean', 'RH_mean', 'AH_mean', 'S5_mean',
             'RMSE', 'MAE', 'JS', 'PredVar', 'CalibGap', 'DriftScore']
descT = pd.DataFrame({v: describe(Wd[v]) for v in desc_vars}).T
descT = descT[['N', 'Mean', 'SD', 'Median', 'IQR', 'Skew', 'Kurt']]
descT.to_csv(RESULTS / 'table_II_descriptives.csv', float_format='%.4f')
print("\nTable II (Descriptives):")
print(descT.round(3))

def cronbach_alpha(items):
    k = items.shape[1]
    var_items = items.var(axis=0, ddof=1).sum()
    var_total = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - var_items / var_total)

composites = {
    'DriftScore (JS, PredVar, CalibGap)': Wd[['JS_z', 'PredVar_z', 'CalibGap_z']].values,
    'PerfIndex (RMSE, MAE)': np.column_stack([
        (Wd['RMSE'] - Wd['RMSE'].mean()) / Wd['RMSE'].std(),
        (Wd['MAE'] - Wd['MAE'].mean()) / Wd['MAE'].std()]),
    'AmbientIndex (T, RH, AH)': np.column_stack([
        (Wd['T_mean']  - Wd['T_mean'].mean())  / Wd['T_mean'].std(),
        (Wd['RH_mean'] - Wd['RH_mean'].mean()) / Wd['RH_mean'].std(),
        (Wd['AH_mean'] - Wd['AH_mean'].mean()) / Wd['AH_mean'].std()]),
}
alpha_rows = []
for name, items in composites.items():
    a = cronbach_alpha(items)
    alpha_rows.append({'Composite': name, 'k_items': items.shape[1], 'alpha': a})
alphaT = pd.DataFrame(alpha_rows)
alphaT.to_csv(RESULTS / 'table_III_reliability.csv', index=False, float_format='%.4f')
print("\nTable III (Reliability):")
print(alphaT.round(3))

IV = ['t_weeks', 'T_mean', 'RH_mean', 'AH_mean', 'S5_mean']
DV = ['RMSE', 'MAE', 'JS', 'PredVar', 'CalibGap', 'DriftScore']

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    num = np.sum((x - xm) * (y - ym))
    den = math.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2))
    r = num / (den + 1e-12)
    n = len(x)
    if abs(r) >= 1.0 - 1e-12:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        df_ = n - 2
        z = t * (1 - 1 / (4 * df_)) / math.sqrt(1 + t * t / (2 * df_))
        p = math.erfc(abs(z) / math.sqrt(2))
    return r, p

corr_rows = []
for iv in IV:
    row = {'IV': iv}
    for dv in DV:
        r, pr = pearson(Wd[iv], Wd[dv])
        row[f'{dv}_r'] = r
        row[f'{dv}_p'] = pr
    corr_rows.append(row)
corrT = pd.DataFrame(corr_rows)
n_tests = len(IV) * len(DV)
for dv in DV:
    corrT[f'{dv}_p_bonf'] = np.clip(corrT[f'{dv}_p'] * n_tests, 0, 1)
corrT.to_csv(RESULTS / 'table_IV_correlations.csv', index=False, float_format='%.4f')
print("\nTable IV (Correlations, r | p_Bonferroni):")
for iv in IV:
    line = [f"{iv:>10s}"]
    for dv in DV:
        r = corrT.loc[corrT['IV'] == iv, f'{dv}_r'].values[0]
        pb = corrT.loc[corrT['IV'] == iv, f'{dv}_p_bonf'].values[0]
        star = '*' if pb < 0.05 else ' '
        line.append(f"{dv}:{r:+.3f}{star}")
    print('  ' + '  '.join(line))

# Hierarchical regression
# Block 1: time; Block 2: + ambient T; Block 3: + humidity (RH, AH); Block 4: + sensor S5 mean
BLOCKS = [
    ['t_weeks'],
    ['t_weeks', 'T_mean'],
    ['t_weeks', 'T_mean', 'RH_mean', 'AH_mean'],
    ['t_weeks', 'T_mean', 'RH_mean', 'AH_mean', 'S5_mean'],
]

def ols(X, y):
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    yhat = Xb @ beta
    resid = y - yhat
    rss = np.sum(resid ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - rss / tss
    dof = n - (p + 1)
    sigma2 = rss / max(dof, 1)
    cov = sigma2 * np.linalg.pinv(Xb.T @ Xb)
    se = np.sqrt(np.diag(cov))
    y_sd = y.std(ddof=1)
    beta_std = np.zeros_like(beta)
    for j in range(p):
        beta_std[j + 1] = beta[j + 1] * X[:, j].std(ddof=1) / y_sd
    t_vals = beta / (se + 1e-12)
    vifs = []
    if p >= 2:
        for j in range(p):
            others = np.delete(np.arange(p), j)
            Xo = np.column_stack([np.ones(n), X[:, others]])
            b, *_ = np.linalg.lstsq(Xo, X[:, j], rcond=None)
            rj = 1 - np.sum((X[:, j] - Xo @ b) ** 2) / np.sum((X[:, j] - X[:, j].mean()) ** 2)
            vifs.append(1 / max(1 - rj, 1e-6))
    else:
        vifs = [np.nan]
    dw = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)
    return dict(beta=beta, beta_std=beta_std, se=se, t=t_vals, r2=r2,
                n=n, p=p, dof=dof, vifs=vifs, dw=dw)

def hier_regression(df_, target):
    rows = []
    prev_r2 = 0.0
    for bi, cols in enumerate(BLOCKS, start=1):
        X = df_[cols].values
        y = df_[target].values
        res = ols(X, y)
        dR2 = res['r2'] - prev_r2
        rows.append({
            'block': bi, 'predictors': ','.join(cols),
            'R2': res['r2'], 'dR2': dR2,
            'DurbinWatson': res['dw'],
            'beta_std': {c: b for c, b in zip(cols, res['beta_std'][1:])},
            't_vals':   {c: t for c, t in zip(cols, res['t'][1:])},
            'VIFs':     {c: v for c, v in zip(cols, res['vifs'])},
        })
        prev_r2 = res['r2']
    return rows

reg_rmse = hier_regression(Wd, 'RMSE')
reg_cal  = hier_regression(Wd, 'CalibGap')

with open(RESULTS / 'table_V_regression.json', 'w') as f:
    json.dump({'RMSE': reg_rmse, 'CalibGap': reg_cal}, f, indent=2, default=str)

print("\nTable V (Hierarchical regression predicting RMSE):")
for r in reg_rmse:
    print(f"  Block {r['block']}: R2={r['R2']:.3f}  dR2={r['dR2']:.3f}  DW={r['DurbinWatson']:.2f}")
    for c, b in r['beta_std'].items():
        print(f"     beta_std({c})={b:+.3f}  t={r['t_vals'][c]:+.2f}  VIF={r['VIFs'][c]:.2f}")

# Repeated-measures ANOVA across 5 epoch bins
n_epochs = 5
Wd['epoch'] = pd.qcut(Wd['t_weeks'], n_epochs, labels=False)
groups = [Wd.loc[Wd['epoch'] == e, 'RMSE'].values for e in range(n_epochs)]
grand = np.concatenate(groups).mean()
ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ssw = sum(np.sum((g - g.mean()) ** 2) for g in groups)
dfb = n_epochs - 1
dfw = sum(len(g) for g in groups) - n_epochs
msb = ssb / dfb
msw = ssw / max(dfw, 1)
F = msb / msw
eta2 = ssb / (ssb + ssw)

def f_cdf(F, d1, d2):
    num = ((1 - 2 / (9 * d2)) * F ** (1 / 3) - (1 - 2 / (9 * d1)))
    den = math.sqrt((2 / (9 * d2)) * F ** (2 / 3) + (2 / (9 * d1)))
    z = num / den
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

p_F = 1 - f_cdf(F, dfb, dfw)
anovaT = pd.DataFrame([{
    'Effect': 'Epoch (within-window factor)',
    'F': F, 'df1': dfb, 'df2': dfw, 'p_approx': p_F, 'eta2_partial': eta2,
}])
anovaT.to_csv(RESULTS / 'table_VI_anova.csv', index=False, float_format='%.4f')
print("\nTable VI (Repeated-measures ANOVA over epochs):")
print(anovaT.round(4).to_string(index=False))

# ---------------------------------------------------------------
# 5. LAGGED REGRESSION: DriftScore(t) -> RMSE(t+k)
# ---------------------------------------------------------------

lags = [1, 2, 3, 4]
lag_rows = []
for k in lags:
    x = Wd['DriftScore'].values[:-k]
    y = Wd['RMSE'].values[k:]
    r, p_ = pearson(x, y)
    x_ = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(x_, y, rcond=None)
    yhat = x_ @ beta
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    lag_rows.append({'lag_weeks': k, 'n_pairs': len(x), 'pearson_r': r,
                     'p_approx': p_, 'R2': r2, 'slope': beta[1]})
lagT = pd.DataFrame(lag_rows)
lagT.to_csv(RESULTS / 'table_VII_leadtime.csv', index=False, float_format='%.4f')
print("\nTable VII (Lagged regression: DriftScore -> future RMSE):")
print(lagT.round(4).to_string(index=False))

# ---------------------------------------------------------------
# 6. FIGURES
# ---------------------------------------------------------------

plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})

fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), dpi=140)
axes[0, 0].plot(Wd['t_weeks'], Wd['RMSE'], color='#b22222')
axes[0, 0].set_title('(a) RMSE'); axes[0, 0].set_xlabel('Weeks since deployment'); axes[0, 0].set_ylabel('RMSE')
axes[0, 1].plot(Wd['t_weeks'], Wd['JS'], color='#1f4e79')
axes[0, 1].set_title('(b) Input-Drift (JS)'); axes[0, 1].set_xlabel('Weeks since deployment'); axes[0, 1].set_ylabel('JS divergence')
axes[1, 0].plot(Wd['t_weeks'], Wd['CalibGap'], color='#7030a0')
axes[1, 0].set_title('(c) Calibration gap'); axes[1, 0].set_xlabel('Weeks since deployment'); axes[1, 0].set_ylabel('|E[y]-E[ŷ]|')
axes[1, 1].plot(Wd['t_weeks'], Wd['DriftScore'], color='#006400')
axes[1, 1].set_title('(d) Composite Drift Score'); axes[1, 1].set_xlabel('Weeks since deployment'); axes[1, 1].set_ylabel('Drift Score (z)')
for ax in axes.ravel():
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / 'fig1_timeseries.png', dpi=180, bbox_inches='tight')
plt.close()

k = 2
x = Wd['DriftScore'].values[:-k]
y = Wd['RMSE'].values[k:]
fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=140)
ax.scatter(x, y, s=14, color='#2e7d32', alpha=0.7)
b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, np.polyval(b, xs), '--', color='k', lw=1.2)
ax.set_xlabel('Drift Score (week t)')
ax.set_ylabel(f'RMSE (week t+{k})')
r = np.corrcoef(x, y)[0, 1]
ax.set_title(f'Lead-time association (k={k}):  r = {r:+.3f}')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / 'fig2_leadtime.png', dpi=180, bbox_inches='tight')
plt.close()

# Lightweight metadata file used by the manuscript builder
meta = {
    'dataset': 'UCI Air Quality (De Vito et al. 2008)',
    'target': TARGET,
    'features': FEAT_COLS,
    'train_weeks': TRAIN_WEEKS,
    'n_hourly_rows_loaded': int(n_before),
    'n_hourly_rows_clean': int(n_after),
    'n_weeks_analyzed': int(len(Wd)),
    'first_timestamp': str(df['Timestamp'].iloc[0]),
    'last_timestamp':  str(df['Timestamp'].iloc[-1]),
    'rmse_first_week': float(Wd['RMSE'].iloc[0]),
    'rmse_last_week':  float(Wd['RMSE'].iloc[-1]),
    'rmse_mean': float(Wd['RMSE'].mean()),
    'rmse_max':  float(Wd['RMSE'].max()),
}
with open(RESULTS / 'meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nArtifacts written to {RESULTS}")
print(sorted(os.listdir(RESULTS)))
