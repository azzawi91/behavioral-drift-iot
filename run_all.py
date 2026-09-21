"""
Run the complete analysis for both deployments and write every table used in
the paper to results/. Usage:

    python run_all.py --uci data/AirQualityUCI.csv --intel data/intel_lab_data.txt

The Intel Lab file is the original `data.txt` from
http://db.csail.mit.edu/labdata/labdata.html (not redistributed here).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import drift_pipeline as dp

ROOT = Path(__file__).resolve().parent
FEATS_UCI = ["S1", "S2", "S3", "S4", "S5", "T", "RH", "AH"]
FEATS_INTEL = ["temp_std", "humid_mean", "humid_std", "light_mean", "light_std",
               "volt_mean", "n_sensors", "hour"]


# ------------------------------------------------------------------ loaders
def load_uci(path):
    raw = pd.read_csv(path, sep=";", decimal=",", na_values=["-200", "-200.0", "-200,0"])
    raw = raw.dropna(axis=1, how="all").dropna(subset=["Date"]).reset_index(drop=True)
    raw["Timestamp"] = pd.to_datetime(raw["Date"] + " " + raw["Time"].str.replace(".", ":", regex=False),
                                      format="%d/%m/%Y %H:%M:%S")
    raw = raw.sort_values("Timestamp").reset_index(drop=True)
    hours = ((raw["Timestamp"] - raw["Timestamp"].iloc[0]).dt.total_seconds() / 3600).astype(int)
    raw["window"] = hours // (24 * 7)
    raw = raw.rename(columns={"PT08.S1(CO)": "S1", "PT08.S2(NMHC)": "S2", "PT08.S3(NOx)": "S3",
                              "PT08.S4(NO2)": "S4", "PT08.S5(O3)": "S5", "C6H6(GT)": "y"})
    n_raw = len(raw)
    df = raw[["Timestamp", "window"] + FEATS_UCI + ["y"]].dropna().reset_index(drop=True)
    return df, {"n_raw": n_raw, "n_clean": len(df),
                "first": str(df["Timestamp"].iloc[0]), "last": str(df["Timestamp"].iloc[-1])}


def load_intel(path, min_motes_per_day=27):
    cols = ["date", "time", "epoch", "moteid", "temp", "humid", "light", "volt"]
    raw = pd.read_csv(path, sep=r"\s+", names=cols, header=None)
    n_raw = len(raw)
    ok = (raw.moteid.between(1, 54) & raw.temp.between(5, 50) & raw.humid.between(0, 105)
          & raw.volt.between(2.0, 3.2) & raw.light.notna())
    d = raw[ok].copy()
    d["ts"] = pd.to_datetime(d["date"] + " " + d["time"], format="mixed")
    d["h"] = d["ts"].dt.floor("h")
    g = d.groupby("h").agg(y=("temp", "mean"), temp_std=("temp", "std"),
                           humid_mean=("humid", "mean"), humid_std=("humid", "std"),
                           light_mean=("light", "mean"), light_std=("light", "std"),
                           volt_mean=("volt", "mean"), n_sensors=("moteid", "nunique")).dropna()
    g["hour"] = g.index.hour
    g["day"] = (g.index.normalize() - g.index.normalize()[0]).days
    # keep the leading run of days on which at least half of the 54 motes still report
    motes_per_day = g.groupby("day")["n_sensors"].mean()
    good = motes_per_day >= min_motes_per_day
    last_day = int((~good).idxmax()) - 1 if (~good).any() else int(good.index.max())
    g = g[g["day"] <= last_day].reset_index().rename(columns={"h": "Timestamp", "day": "window"})
    used = d[(d["h"].dt.normalize() - d["h"].dt.normalize().min()).dt.days <= last_day]
    return g, {"n_raw": n_raw, "n_clean_readings_whole_record": int(ok.sum()),
               "n_clean_readings_used": int(len(used)), "n_motes_used": int(used["moteid"].nunique()),
               "n_out_of_range_temp": int((~raw.temp.between(5, 50) & raw.temp.notna()).sum()),
               "n_out_of_range_volt": int((~raw.volt.between(2.0, 3.2) & raw.volt.notna()).sum()),
               "n_hourly": len(g),
               "n_days": last_day + 1, "first": str(g["Timestamp"].iloc[0]), "last": str(g["Timestamp"].iloc[-1]),
               "motes_first_day": float(motes_per_day.iloc[0]), "motes_last_day": float(motes_per_day.loc[last_day])}


# ------------------------------------------------------------------ analysis
def analyse(name, df, feats, train_windows, min_samples, covariates, blocks, burn_in, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    Wd, samples, FJ = dp.window_metrics(df, feats, "y", "window", train_windows, min_samples, covariates)
    Wd.to_csv(outdir / "window_metrics.csv", index=False)
    pd.DataFrame(FJ, columns=feats).to_csv(outdir / "per_feature_jsd.csv", index=False)
    S = {"n_windows": len(Wd), "rmse_first": Wd.RMSE.iloc[0], "rmse_last": Wd.RMSE.iloc[-1],
         "rmse_factor": Wd.RMSE.iloc[-1] / Wd.RMSE.iloc[0], "rmse_mean": Wd.RMSE.mean(),
         "rmse_sd": Wd.RMSE.std(), "rmse_max": Wd.RMSE.max(), "rmse_argmax_t": int(Wd.RMSE.idxmax())}

    desc_vars = ["RMSE", "MAE", "JSD", "CAL", "SPD", "DS"] + covariates
    desc = pd.DataFrame({v: dp.describe(Wd[v]) for v in desc_vars}).T
    desc.to_csv(outdir / "descriptives.csv", float_format="%.4f")

    S["cronbach_alpha"] = dp.cronbach_alpha(Wd[["JSD_z", "SPD_z", "CAL_z"]].values)
    S["component_corr"] = Wd[["JSD", "SPD", "CAL"]].corr().round(3).to_dict()

    ivs, dvs = ["t"] + covariates, ["RMSE", "JSD", "SPD", "CAL", "DS"]
    rows = []
    for iv in ivs:
        for dv in dvs:
            r, p = dp.pearson(Wd[iv], Wd[dv])
            rows.append({"IV": iv, "DV": dv, "r": r, "p": p})
    corr = pd.DataFrame(rows)
    corr["p_bonf"] = np.clip(corr["p"] * len(corr), 0, 1)
    corr.to_csv(outdir / "correlations.csv", index=False, float_format="%.4f")
    S["n_corr_tests"] = len(corr)

    S["hier_regression"] = dp.hier_regression(Wd, blocks)
    S["anova"] = dp.epoch_anova(Wd)

    lt = dp.lead_time(Wd["DS"], Wd["RMSE"])
    lt.to_csv(outdir / "lead_time.csv", index=False, float_format="%.4f")

    # ablation and reactive baselines at k = 1
    base = dp.baseline_window_stats(samples, "window", train_windows, Wd["window"].values)
    signals = {"DS": Wd["DS"], "JSD": Wd["JSD"], "SPD": Wd["SPD"], "CAL": Wd["CAL"],
               "JSD+SPD (label-free)": (Wd["JSD_z"] + Wd["SPD_z"]) / 2,
               "ADWIN": base["ADWIN"], "Page-Hinkley": base["PH"], "DDM": base["DDM"],
               "Persistence (RMSE_t)": Wd["RMSE"]}
    ab = []
    for k, s in signals.items():
        row = dp.lead_time(s, Wd["RMSE"], lags=(1,)).iloc[0].to_dict()
        row["signal"] = k
        ab.append(row)
    pd.DataFrame(ab)[["signal", "n", "r", "R2", "p", "t_NW", "p_NW"]].to_csv(
        outdir / "ablation_baselines.csv", index=False, float_format="%.4f")

    # does DS add information beyond persistence?
    x, y, z = Wd["DS"].values[:-1], Wd["RMSE"].values[1:], Wd["RMSE"].values[:-1]
    S["partial_r_DS_given_RMSE"], S["partial_p_DS_given_RMSE"] = dp.partial_corr(x, y, z)
    S["partial_r_RMSE_given_DS"], S["partial_p_RMSE_given_DS"] = dp.partial_corr(z, y, x)
    S["r_DS_RMSE_same_window"] = dp.pearson(Wd["DS"], Wd["RMSE"])[0]
    S["rmse_lag1_autocorr"] = dp.pearson(z, y)[0]

    # causal (burn-in) standardization
    ds_c, rm_c = dp.causal_ds(Wd, burn_in)
    ltc = dp.lead_time(ds_c, rm_c)
    ltc.to_csv(outdir / "lead_time_causal.csv", index=False, float_format="%.4f")
    S["burn_in"] = burn_in

    # retrospective DS restricted to the same post-burn-in windows (isolates the role of the early ramp-up)
    S["retro_DS_post_burn_in_k1"] = dict(zip(("r", "p"), dp.pearson(Wd["DS"].values[burn_in:-1], Wd["RMSE"].values[burn_in + 1:])))

    # trend robustness: remove a linear time trend from signal and RMSE before correlating (k = 1)
    detr = lambda v: dp.ols(Wd["t"].values.reshape(-1, 1).astype(float), np.asarray(v, float))["resid"]
    S["detrended_k1"] = {k: dict(zip(("r", "p"), dp.pearson(detr(Wd[k])[:-1], detr(Wd["RMSE"])[1:])))
                         for k in ["DS", "JSD", "SPD", "CAL", "RMSE"]}
    S["first_difference_k1"] = {k: dict(zip(("r", "p"), dp.pearson(np.diff(Wd[k])[:-1], np.diff(Wd["RMSE"])[1:])))
                                for k in ["DS", "JSD", "RMSE"]}
    S["covariate_time_corr"] = {c: dp.pearson(Wd["t"], Wd[c])[0] for c in covariates}
    S["covariate_first_last"] = {c: [float(Wd[c].iloc[0]), float(Wd[c].iloc[-1])] for c in covariates}
    S["peak_DS"] = {"t": int(Wd["DS"].idxmax()), "DS": float(Wd["DS"].max()),
                    "per_feature_jsd": dict(zip(feats, FJ[int(Wd["DS"].idxmax())]))}
    S["short_windows"] = Wd.loc[Wd["n_samples"] < (24 if name == "intel" else 168 * 0.5), ["t", "n_samples"]].values.tolist()

    # early-warning flag (retrospective threshold mu + 0.5 sd)
    thr = Wd["DS"].mean() + 0.5 * Wd["DS"].std()
    flag = (Wd["DS"].values[:-1] > thr)
    nxt = Wd["RMSE"].values[1:]
    u = dp.stats.mannwhitneyu(nxt[flag], nxt[~flag], alternative="greater")
    S["flag"] = {"threshold": thr, "n_flagged": int(flag.sum()), "n_unflagged": int((~flag).sum()),
                 "next_rmse_flagged": float(nxt[flag].mean()), "next_rmse_unflagged": float(nxt[~flag].mean()),
                 "mannwhitney_p": float(u.pvalue)}
    return Wd, samples, FJ, S


def seed_sweep(df, feats, train_windows, min_samples, covariates, n=10):
    rows = []
    for s in range(n):
        Wd, _, _ = dp.window_metrics(df, feats, "y", "window", train_windows, min_samples, covariates,
                                     init_seed=s, shuffle_seed=100 + s)
        lt = dp.lead_time(Wd["DS"], Wd["RMSE"])
        rows.append({"seed": s, "rmse_first": Wd.RMSE.iloc[0], "rmse_last": Wd.RMSE.iloc[-1],
                     "rmse_mean": Wd.RMSE.mean(), **{f"r_k{int(k)}": r for k, r in zip(lt.k, lt.r)}})
    return pd.DataFrame(rows)


def sweep_summary(sw):
    ratio = sw["rmse_mean"] / sw["rmse_first"]
    lf = sw["rmse_last"] / sw["rmse_first"]
    return {"n_seeds": len(sw), "r_k1_mean": sw["r_k1"].mean(), "r_k1_min": sw["r_k1"].min(), "r_k1_max": sw["r_k1"].max(),
            "r_k1_n_positive": int((sw["r_k1"] > 0).sum()),
            "mean_over_first_rmse_min": ratio.min(), "mean_over_first_rmse_max": ratio.max(),
            "mean_over_first_rmse_median": ratio.median(),
            "last_over_first_rmse_min": lf.min(), "last_over_first_rmse_max": lf.max()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci", default=str(ROOT / "data" / "AirQualityUCI.csv"))
    ap.add_argument("--intel", default=str(ROOT / "data" / "intel_lab_data.txt"))
    a = ap.parse_args()
    res = ROOT / "results"
    summary = {}

    uci, meta = load_uci(a.uci)
    blocks = [("Time", ["t"]), ("+ Temperature", ["t", "T"]), ("+ Humidity (RH, AH)", ["t", "T", "RH", "AH"])]
    Wd, samples, FJ, S = analyse("uci", uci, FEATS_UCI, 4, 80, ["T", "RH", "AH"], blocks, 12, res / "uci")
    # post hoc: JSD is an unsigned distance, so relate it to the absolute departure of each
    # ambient covariate from its mean in the reference (training) weeks
    ref_mean = uci[uci["window"] < 4][["T", "RH", "AH"]].mean()
    S["ref_covariate_means"] = ref_mean.to_dict()
    S["jsd_vs_abs_departure"] = {c: dict(zip(("r", "p"), dp.pearson(Wd["JSD"], (Wd[c] - ref_mean[c]).abs())))
                                 for c in ["T", "RH", "AH"]}
    S["n_train_samples"] = int((uci["window"] < 4).sum())
    sw = seed_sweep(uci, FEATS_UCI, 4, 80, ["T", "RH", "AH"])
    sw.to_csv(res / "uci" / "seed_sweep.csv", index=False, float_format="%.4f")
    S["seed_sweep"] = sweep_summary(sw)
    summary["uci"] = {**meta, **S}

    if Path(a.intel).exists():
        intel, meta = load_intel(a.intel)
        cov = ["volt_mean", "humid_mean", "light_mean", "n_sensors"]
        blocks = [("Time", ["t"]), ("+ Ambient (humidity, light)", ["t", "humid_mean", "light_mean"]),
                  ("+ Device (voltage, active motes)", ["t", "humid_mean", "light_mean", "volt_mean", "n_sensors"])]
        Wd, samples, FJ, S = analyse("intel", intel, FEATS_INTEL, 3, 10, cov, blocks, 7, res / "intel")
        intel.to_csv(res / "intel" / "hourly_aggregates.csv", index=False)
        S["n_train_samples"] = int((intel["window"] < 3).sum())
        S["cal_rmse_corr"] = dp.pearson(Wd["CAL"], Wd["RMSE"])[0]
        sw = seed_sweep(intel, FEATS_INTEL, 3, 10, cov)
        sw.to_csv(res / "intel" / "seed_sweep.csv", index=False, float_format="%.4f")
        S["seed_sweep"] = sweep_summary(sw)
        summary["intel"] = {**meta, **S}
    else:
        print(f"Intel Lab file not found at {a.intel}; skipping Dataset 2.")

    dp.save_json(summary, res / "summary.json")
    print("done ->", res)


if __name__ == "__main__":
    main()
