"""
Early warning of behavioral drift in edge-deployed deep learning for IoT.

Shared pipeline used for both deployments (UCI Air Quality, Intel Berkeley Lab).
Everything is plain NumPy / pandas; SciPy is used only for exact p-values.

The UCI branch is numerically identical to the original `run_analysis_real.py`
(same seeds, same initialization, same update order), so the weekly metrics and
the lead-time table reproduce the original run bit-for-bit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BINS = 20          # histogram bins for JSD (K in the paper)
EPOCHS = 200
BATCH = 128
LR = 2e-3
INIT_SEED = 0      # weight initialization
SHUFFLE_SEED = 42  # mini-batch shuffling


# --------------------------------------------------------------------------
# Compact MLP 8 -> 16 -> 8 -> 1 (289 parameters), pure NumPy
# --------------------------------------------------------------------------
def init_mlp(d_in=8, h1=16, h2=8, d_out=1, seed=INIT_SEED):
    r = np.random.default_rng(seed)
    return {
        "W1": r.normal(0, np.sqrt(2 / d_in), (d_in, h1)), "b1": np.zeros(h1),
        "W2": r.normal(0, np.sqrt(2 / h1), (h1, h2)), "b2": np.zeros(h2),
        "W3": r.normal(0, np.sqrt(2 / h2), (h2, d_out)), "b3": np.zeros(d_out),
    }


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(0, z1)
    z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(0, z2)
    z3 = a2 @ p["W3"] + p["b3"]
    return z3.squeeze(-1), (X, z1, a1, z2, a2, z3)


def backward(p, cache, y):
    X, z1, a1, z2, a2, z3 = cache
    n = X.shape[0]
    dz3 = (z3.squeeze(-1) - y).reshape(-1, 1) / n
    g = {"W3": a2.T @ dz3, "b3": dz3.sum(0)}
    dz2 = (dz3 @ p["W3"].T) * (z2 > 0)
    g["W2"] = a1.T @ dz2; g["b2"] = dz2.sum(0)
    dz1 = (dz2 @ p["W2"].T) * (z1 > 0)
    g["W1"] = X.T @ dz1; g["b1"] = dz1.sum(0)
    return g


def adam_step(p, g, state, lr, t, b1=0.9, b2=0.999, eps=1e-8):
    for k in p:
        if k not in state:
            state[k] = {"m": np.zeros_like(p[k]), "v": np.zeros_like(p[k])}
        s = state[k]
        s["m"] = b1 * s["m"] + (1 - b1) * g[k]
        s["v"] = b2 * s["v"] + (1 - b2) * (g[k] ** 2)
        p[k] -= lr * (s["m"] / (1 - b1 ** t)) / (np.sqrt(s["v"] / (1 - b2 ** t)) + eps)


def train_mlp(Xtr, ytr, init_seed=INIT_SEED, shuffle_seed=SHUFFLE_SEED):
    rng = np.random.default_rng(shuffle_seed)
    p, state = init_mlp(d_in=Xtr.shape[1], seed=init_seed), {}
    for epoch in range(1, EPOCHS + 1):
        idx = rng.permutation(len(Xtr))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            _, cache = forward(p, Xtr[b])
            adam_step(p, backward(p, cache, ytr[b]), state, lr=LR, t=epoch)
    return p


# --------------------------------------------------------------------------
# Drift metrics
# --------------------------------------------------------------------------
def hist_pdf(x, edges):
    h, _ = np.histogram(x, bins=edges)
    h = h + 1.0                      # Laplace smoothing
    return h / h.sum()


def js_div(p_, q_):
    m = 0.5 * (p_ + q_)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p_, m) + 0.5 * kl(q_, m)


def window_metrics(df, feat_cols, target, window_col, train_windows, min_samples,
                   covariates, init_seed=INIT_SEED, shuffle_seed=SHUFFLE_SEED):
    """Train on the reference windows, freeze, and compute per-window metrics.

    Returns (per-window DataFrame for deployment windows, per-sample DataFrame,
    per-window per-feature JSD array).
    """
    X = df[feat_cols].values.astype(float)
    y = df[target].values.astype(float)
    ref = (df[window_col] < train_windows).values
    mu, sd = X[ref].mean(0), X[ref].std(0) + 1e-9
    Xn = (X - mu) / sd
    ymu, ysd = y[ref].mean(), y[ref].std() + 1e-9
    p = train_mlp(Xn[ref], (y[ref] - ymu) / ysd, init_seed, shuffle_seed)

    out = df.copy()
    out["yhat"] = forward(p, Xn)[0] * ysd + ymu
    out["err"] = out[target] - out["yhat"]

    edges = [np.linspace(Xn[:, j].min(), Xn[:, j].max(), BINS + 1) for j in range(Xn.shape[1])]
    ref_h = [hist_pdf(Xn[ref, j], edges[j]) for j in range(Xn.shape[1])]
    ref_spread = out.loc[ref, "yhat"].std()

    rows, feat_js = [], []
    for w in sorted(out[window_col].unique()):
        m = (out[window_col] == w).values
        if m.sum() < min_samples:
            continue
        js_j = [js_div(hist_pdf(Xn[m, j], edges[j]), ref_h[j]) for j in range(Xn.shape[1])]
        e = out.loc[m, "err"].values
        row = {
            "window": int(w), "n_samples": int(m.sum()),
            "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "JSD": float(np.mean(js_j)),
            "SPD": float(abs(out.loc[m, "yhat"].std() - ref_spread)),
            "CAL": float(abs(out.loc[m, target].mean() - out.loc[m, "yhat"].mean())),
        }
        for c in covariates:
            row[c] = float(out.loc[m, c].mean())
        rows.append(row); feat_js.append(js_j)
    W = pd.DataFrame(rows)
    FJ = np.array(feat_js)

    # Composite Drift Score: retrospective z-scores over the whole record
    for c in ["JSD", "SPD", "CAL"]:
        W[c + "_z"] = (W[c] - W[c].mean()) / W[c].std()
    W["DS"] = W[["JSD_z", "SPD_z", "CAL_z"]].mean(axis=1)

    keep = (W["window"] >= train_windows).values
    Wd = W[keep].reset_index(drop=True)
    Wd["t"] = Wd["window"] - train_windows
    return Wd, out, FJ[keep]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def describe(x):
    x = np.asarray(x, float)
    q1, q3 = np.percentile(x, [25, 75])
    return {"Mean": x.mean(), "SD": x.std(ddof=1), "Median": np.median(x),
            "IQR": q3 - q1, "Skew": stats.skew(x)}


def cronbach_alpha(items):
    k = items.shape[1]
    return (k / (k - 1)) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1))


def pearson(x, y):
    r, p = stats.pearsonr(np.asarray(x, float), np.asarray(y, float))
    return float(r), float(p)


def ols(X, y):
    n, k = X.shape
    Xb = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
    resid = y - Xb @ beta
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
    beta_std = beta[1:] * X.std(0, ddof=1) / y.std(ddof=1)
    dw = np.sum(np.diff(resid) ** 2) / (resid @ resid)
    return {"beta": beta, "beta_std": beta_std, "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k - 1),
            "dw": dw, "resid": resid, "Xb": Xb}


def newey_west_t(x, y, lags=4):
    """Slope t statistic with Newey-West (Bartlett) HAC standard errors."""
    res = ols(np.asarray(x, float).reshape(-1, 1), np.asarray(y, float))
    Xb, u = res["Xb"], res["resid"]
    n = len(u)
    Z = Xb * u[:, None]
    S = Z.T @ Z
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Z[l:].T @ Z[:-l]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(Xb.T @ Xb)
    cov = XtX_inv @ S @ XtX_inv * n / (n - 2)
    t = res["beta"][1] / math.sqrt(cov[1, 1])
    return float(t), float(2 * stats.t.sf(abs(t), n - 2))


def vifs(X, cols):
    if X.shape[1] < 2:
        return {}
    out = {}
    for j, c in enumerate(cols):
        r2 = ols(np.delete(X, j, axis=1), X[:, j])["r2"]
        out[c] = 1.0 / max(1 - r2, 1e-9)
    return out


def hier_regression(Wd, blocks, target="RMSE"):
    rows, prev = [], 0.0
    for name, cols in blocks:
        res = ols(Wd[cols].values, Wd[target].values)
        rows.append({"block": name, "predictors": cols, "R2": res["r2"], "adj_R2": res["adj_r2"],
                     "dR2": res["r2"] - prev, "DW": res["dw"],
                     "beta_std": dict(zip(cols, res["beta_std"])), "VIF": vifs(Wd[cols].values, cols)})
        prev = res["r2"]
    return rows


def epoch_anova(Wd, n_epochs=5):
    ep = pd.qcut(Wd["t"], n_epochs, labels=False)
    groups = [Wd.loc[ep == e, "RMSE"].values for e in range(n_epochs)]
    F, p = stats.f_oneway(*groups)
    grand = Wd["RMSE"].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    return {"F": float(F), "df1": n_epochs - 1, "df2": len(Wd) - n_epochs, "p": float(p),
            "eta2": float(ssb / (ssb + ssw)), "sizes": [len(g) for g in groups],
            "epoch_means": [float(g.mean()) for g in groups]}


def lead_time(signal, rmse, lags=(1, 2, 3, 4), hac_lags=4):
    rows = []
    s, r_ = np.asarray(signal, float), np.asarray(rmse, float)
    for k in lags:
        x, y = s[:-k], r_[k:]
        r, p = pearson(x, y)
        slope = np.polyfit(x, y, 1)[0]
        t_nw, p_nw = newey_west_t(x, y, hac_lags)
        rows.append({"k": k, "n": len(x), "r": r, "R2": r * r, "slope": float(slope),
                     "p": p, "t_NW": t_nw, "p_NW": p_nw})
    return pd.DataFrame(rows)


def partial_corr(x, y, z):
    """Correlation of x and y after removing the linear effect of z from both."""
    rx = ols(np.asarray(z, float).reshape(-1, 1), np.asarray(x, float))["resid"]
    ry = ols(np.asarray(z, float).reshape(-1, 1), np.asarray(y, float))["resid"]
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(rx)
    t = r * math.sqrt((n - 3) / (1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), n - 3))


def causal_ds(Wd, burn_in):
    """Drift Score with z-score constants estimated from the first `burn_in`
    deployment windows only; returned for the remaining windows."""
    z = []
    for c in ["JSD", "SPD", "CAL"]:
        mu, sd = Wd[c].iloc[:burn_in].mean(), Wd[c].iloc[:burn_in].std()
        z.append((Wd[c] - mu) / sd)
    return (sum(z) / 3).iloc[burn_in:].values, Wd["RMSE"].iloc[burn_in:].values


# --------------------------------------------------------------------------
# Reactive baselines, run on the per-sample absolute-error stream (fully labeled)
# --------------------------------------------------------------------------
def page_hinkley_stat(x, delta):
    """Continuous Page-Hinkley statistic PH_t = m_t - min_{s<=t} m_s."""
    mean, m, mmin, out = 0.0, 0.0, 0.0, np.empty(len(x))
    for i, v in enumerate(x, 1):
        mean += (v - mean) / i
        m += v - mean - delta
        mmin = min(mmin, m)
        out[i - 1] = m - mmin
    return out


def ddm_stat(b):
    """Continuous DDM statistic (p_t + s_t - p_min - s_min) / s_min on a binary error stream."""
    p, pmin, smin, out = 0.0, np.inf, np.inf, np.zeros(len(b))
    for i, v in enumerate(b, 1):
        p += (v - p) / i
        s = math.sqrt(max(p * (1 - p), 1e-12) / i)
        if i >= 30:
            if p + s < pmin + smin:
                pmin, smin = p, s
            out[i - 1] = (p + s - pmin - smin) / smin
    return out


def adwin_stat(x, delta=0.002, step=24, max_window=24 * 7 * 12):
    """ADWIN on a [0,1] stream. Returns, per sample, the largest ratio
    |mean_old - mean_new| / eps_cut over the examined cuts (>= 1 means a cut is
    detected), evaluated before the window is shrunk."""
    win, out = [], np.zeros(len(x))
    for i, v in enumerate(x):
        win.append(v)
        if len(win) > max_window:
            win = win[-max_window:]
        if i % step or len(win) < 2 * step:
            continue                 # statistic is only evaluated every `step` samples
        best = 0.0
        while True:
            a = np.asarray(win); n = len(a); cs = np.cumsum(a); tot = cs[-1]
            var = a.var()
            shrink, best_here = False, 0.0
            for c in range(step, n - step + 1, step):
                n0, n1 = c, n - c
                m_h = 1.0 / (1.0 / n0 + 1.0 / n1)
                dp = math.log(2 * math.log(n) / delta)
                eps = math.sqrt(2.0 / m_h * var * dp) + 2.0 / (3 * m_h) * dp
                ratio = abs(cs[c - 1] / n0 - (tot - cs[c - 1]) / n1) / eps
                best_here = max(best_here, ratio)
            best = max(best, best_here)
            if best_here >= 1.0 and len(win) > 2 * step:
                win = win[step:]
                shrink = True
            if not shrink:
                break
        out[i] = best
    return out


def baseline_window_stats(samples, window_col, train_windows, windows):
    a = samples["err"].abs().values
    ref = (samples[window_col] < train_windows).values
    ph = page_hinkley_stat(a, delta=0.5 * a[ref].std())
    ddm = ddm_stat((a > np.percentile(a[ref], 90)).astype(float))
    adw = adwin_stat(np.clip(a / np.percentile(a, 99.5), 0, 1))
    s = samples.assign(PH=ph, DDM=ddm, ADWIN=adw)
    g = s.groupby(window_col).agg(PH=("PH", "last"), DDM=("DDM", "last"), ADWIN=("ADWIN", "max"))
    return g.loc[windows].reset_index(drop=True)


# --------------------------------------------------------------------------
def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, default=conv))
