"""Regenerate Figs. 3-7 of the paper from results/ (run after run_all.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, OUT = ROOT / "results", ROOT / "figures"
OUT.mkdir(exist_ok=True)
S = json.loads((RES / "summary.json").read_text())

# colour-blind-safe palette (Okabe-Ito), one hue per role
BLUE, VERM, GREEN, PURPLE, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#7B4EA3", "#8A8A8A", "#222222"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.2, "axes.labelsize": 7,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#444444", "ytick.color": "#444444", "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 1.1, "legend.frameon": False, "legend.fontsize": 6.5, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel(ax, x, y, color, title, ylabel, xlabel, trend=True):
    ax.plot(x, y, color=color)
    if trend:
        b = np.polyfit(x, y, 1)
        ax.plot(x, np.polyval(b, x), ls="--", lw=0.8, color=GREY)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)


def metrics_figure(ds, name, xlabel, unit, extra):
    W = pd.read_csv(RES / ds / "window_metrics.csv")
    x = W["t"] + 1
    fig, axes = plt.subplots(3, 2, figsize=(4.9, 5.0))
    specs = [("RMSE", VERM, "a  RMSE", f"RMSE ({unit})"), ("JSD", BLUE, "b  Input shift (JSD)", "JSD (nats)"),
             ("SPD", PURPLE, "c  Prediction-spread deviation", f"SPD ({unit})"),
             ("CAL", GREEN, "d  Calibration gap", f"CAL ({unit})"), ("DS", INK, "e  Composite Drift Score", "DS (z)")]
    for ax, (c, col, ttl, yl) in zip(axes.ravel(), specs):
        panel(ax, x, W[c], col, ttl, yl, xlabel)
    thr = S[ds]["flag"]["threshold"]
    axes[2, 0].axhline(thr, color=GREY, ls=":", lw=1)
    col, ttl, yl = extra
    panel(axes[2, 1], x, W[col], GREY, ttl, yl, xlabel, trend=False)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, name)


def lagged_figure():
    W = pd.read_csv(RES / "uci" / "window_metrics.csv")
    L = pd.read_csv(RES / "uci" / "lead_time.csv")
    fig, axes = plt.subplots(2, 2, figsize=(4.9, 3.6), sharey=True, sharex=True)
    for ax, (_, row) in zip(axes.ravel(), L.iterrows()):
        k = int(row.k)
        xs, ys = W["DS"].values[:-k], W["RMSE"].values[k:]
        ax.scatter(xs, ys, s=11, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.4)
        g = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(g, np.polyval(np.polyfit(xs, ys, 1), g), color=INK, lw=1)
        ax.set_title(f"k = {k}   r = {row.r:+.2f}", loc="left", fontweight="bold")
    for ax in axes[1]: ax.set_xlabel(r"$\mathrm{DS}_t$")
    for ax in axes[:, 0]: ax.set_ylabel(r"$\mathrm{RMSE}_{t+k}$ (µg/m³)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig4_uci_lagged")


def warning_figure():
    W = pd.read_csv(RES / "uci" / "window_metrics.csv")
    FJ = pd.read_csv(RES / "uci" / "per_feature_jsd.csv")
    pk, thr = S["uci"]["peak_DS"]["t"], S["uci"]["flag"]["threshold"]
    fig, axes = plt.subplots(1, 3, figsize=(4.9, 1.9), gridspec_kw={"width_ratios": [1.15, 1.5, 0.95]})
    names = ["S1", "S2", "S3", "S4", "S5", "T", "RH", "AH"]
    axes[0].bar(names, FJ.iloc[pk].values, color=BLUE, width=0.62)
    axes[0].tick_params(axis="x", labelsize=5.5)
    axes[0].axhline(FJ.iloc[pk].mean(), color=GREY, ls=":", lw=1)
    axes[0].set_title("a  JSD by feature", loc="left", fontweight="bold")
    axes[0].set_ylabel("JSD (nats)"); axes[0].grid(axis="x", visible=False)
    x = W["t"] + 1
    flag = W["DS"] > thr
    axes[1].plot(x, W["DS"], color=INK)
    axes[1].scatter(x[flag], W["DS"][flag], s=14, color=VERM, zorder=3, label="flagged week")
    axes[1].axhline(thr, color=GREY, ls=":", lw=1)
    axes[1].set_title("b  DS and flags", loc="left", fontweight="bold")
    axes[1].set_xlabel("Deployment week"); axes[1].set_ylabel("DS (z)")
    f = flag.values[:-1]; nxt = W["RMSE"].values[1:]
    rng = np.random.default_rng(1)
    for i, (m, col) in enumerate([(~f, GREY), (f, VERM)]):
        axes[2].scatter(i + rng.uniform(-0.12, 0.12, m.sum()), nxt[m], s=10, color=col, alpha=0.75, edgecolor="white", linewidth=0.3)
        axes[2].hlines(nxt[m].mean(), i - 0.25, i + 0.25, color=INK, lw=1.5)
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["no flag", "flag"])
    axes[2].set_xlim(-0.5, 1.5); axes[2].grid(axis="x", visible=False)
    axes[2].set_title("c  Next-week RMSE", loc="left", fontweight="bold"); axes[2].set_ylabel("RMSE (µg/m³)")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig5_uci_warning")


def cross_figure():
    U = pd.read_csv(RES / "uci" / "window_metrics.csv")
    I = pd.read_csv(RES / "intel" / "window_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(4.9, 1.9))
    axes[0].plot(U["t"] / U["t"].max(), U["RMSE"] / U["RMSE"].iloc[0], color=VERM, label="UCI")
    axes[0].plot(I["t"] / I["t"].max(), I["RMSE"] / I["RMSE"].iloc[0], color=BLUE, ls="--", label="Intel Lab")
    axes[0].axhline(1, color=GREY, lw=0.8, ls=":")
    axes[0].set_title("a  Normalized RMSE", loc="left", fontweight="bold")
    axes[0].set_xlabel("Normalized deployment time"); axes[0].set_ylabel(r"$\mathrm{RMSE}_t/\mathrm{RMSE}_1$"); axes[0].legend(loc="upper left")
    U = U.assign(dT=(U["T"] - S["uci"]["ref_covariate_means"]["T"]).abs())
    for ax, (D, xc, yc, col, ttl, xl, yl) in zip(axes[1:], [
            (U, "dT", "JSD", VERM, "b  UCI", r"$|T-T_{\mathrm{ref}}|$ (°C)", "JSD (nats)"),
            (I, "volt_mean", "JSD", BLUE, "c  Intel Lab", "Mean battery voltage (V)", "JSD (nats)")]):
        ax.scatter(D[xc], D[yc], s=12, color=col, alpha=0.8, edgecolor="white", linewidth=0.4)
        g = np.linspace(D[xc].min(), D[xc].max(), 50)
        ax.plot(g, np.polyval(np.polyfit(D[xc], D[yc], 1), g), color=INK, lw=1)
        r = np.corrcoef(D[xc], D[yc])[0, 1]
        ax.set_title(ttl, loc="left", fontweight="bold"); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.text(0.97, 0.05 if r > 0 else 0.9, f"r = {r:+.2f}".replace("-", "\u2212"), transform=ax.transAxes, ha="right", fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig7_cross_dataset")


if __name__ == "__main__":
    metrics_figure("uci", "fig3_uci_metrics", "Deployment week", "µg/m³", ("T", "f  Ambient temperature", "T (°C)"))
    lagged_figure()
    warning_figure()
    metrics_figure("intel", "fig6_intel_metrics", "Deployment day", "°C", ("volt_mean", "f  Mean battery voltage", "Voltage (V)"))
    cross_figure()
    print("figures ->", OUT)
