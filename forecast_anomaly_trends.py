#!/usr/bin/env python3
"""Forecast anomaly trends using EWMA control charts and rolling anomaly rates.

For each station CSV (sorted by DATE/TIME), the script:
1. Computes EWMA and 3-sigma control limits for every measurement column
2. Plots EWMA charts for the N most-anomalous columns
3. Computes a rolling anomaly rate from existing anomaly flags
4. Outputs a trend summary table

The data is too limited (single day, small-N stations) for ARIMA/Prophet or
ML classifiers.  This script instead reports whether recent units are drifting
or showing rising anomaly rates — simple, honest, and interpretable.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METADATA_COLUMNS = {"JSN", "DATE", "TIME", "Lab#", "File Name"}
DEFAULT_INPUTS = (Path("data_stn08.csv"), Path("data_stn18.csv"), Path("data_biwpc.csv"))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "column"


def measurement_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name for name in frame.columns
        if name not in METADATA_COLUMNS and not name.startswith("Unnamed:")
    ]


def parse_datetime(frame: pd.DataFrame) -> pd.Series:
    """Combine DATE and TIME columns into a single datetime series."""
    dt_str = frame["DATE"].astype(str) + " " + frame["TIME"].astype(str)
    return pd.to_datetime(dt_str, errors="coerce")


def ewma_control_chart(
    values: np.ndarray,
    lambdas: tuple[float, ...] = (0.05, 0.1, 0.2),
    sigma_multiplier: float = 3.0,
) -> dict:
    """Compute EWMA and control limits for multiple lambda values.

    Returns dict with keys like 'ewma_{lam}', 'ucl_{lam}', 'lcl_{lam}'.
    """
    n = len(values)
    result = {}
    for lam in lambdas:
        ewma = np.full(n, np.nan)
        if n == 0:
            continue
        ewma[0] = values[0]
        for i in range(1, n):
            ewma[i] = lam * values[i] + (1 - lam) * ewma[i - 1]
        sigma = np.std(values, ddof=1)
        # Steady-state control limits for EWMA
        se = sigma * np.sqrt(lam / (2 - lam))
        ucl = np.full(n, np.mean(values) + sigma_multiplier * se)
        lcl = np.full(n, np.mean(values) - sigma_multiplier * se)
        # Widen limits for the first few points (transient)
        for i in range(1, n):
            scale = np.sqrt(lam * (1 - (1 - lam) ** (2 * i)) / (2 - lam))
            transient_se = sigma * scale
            ucl[i] = np.mean(values) + sigma_multiplier * transient_se
            lcl[i] = np.mean(values) - sigma_multiplier * transient_se
        result[f"ewma_{lam}"] = ewma
        result[f"ucl_{lam}"] = ucl
        result[f"lcl_{lam}"] = lcl
    return result


def rolling_anomaly_rate(
    anomaly_flags: pd.Series,
    window: int = 50,
    min_periods: int = 10,
) -> pd.Series:
    rolling = anomaly_flags.rolling(window=window, min_periods=min_periods)
    return rolling.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument(
        "--anomaly-dir",
        type=Path,
        default=Path("anomaly_analysis"),
        help="Directory with unit_by_column_anomalies.csv and unit_level_composite_anomalies.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("forecast_output"))
    parser.add_argument(
        "--ewma-lambdas",
        type=float,
        nargs="*",
        default=[0.05, 0.1, 0.2],
        help="EWMA smoothing parameters to try (default: 0.05 0.1 0.2)",
    )
    parser.add_argument(
        "--sigma-multiplier",
        type=float,
        default=3.0,
        help="Control-limit multiplier (default: 3.0)",
    )
    parser.add_argument(
        "--top-n-columns",
        type=int,
        default=10,
        help="Number of highest-anomaly-rate columns to plot EWMA for (default: 10)",
    )
    parser.add_argument(
        "--anomaly-rate-window",
        type=int,
        default=50,
        help="Rolling window for anomaly-rate trend (default: 50)",
    )
    parser.add_argument(
    "--profile",
    type=Path,
    default=Path("distribution_profiles/distribution_profile_all_columns.csv"),
    help="Distribution profile CSV (only profiled columns are analysed).",
)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.profile.exists():
        parser.error(f"Profile not found: {args.profile}")
    profile = pd.read_csv(args.profile)
    per_column = pd.read_csv(args.anomaly_dir / "unit_by_column_anomalies.csv")
    unit_level = pd.read_csv(args.anomaly_dir / "unit_level_composite_anomalies.csv")

    all_summaries: list[dict] = []

    for path in args.inputs:
        frame = pd.read_csv(path)
        station = path.stem
        columns = measurement_columns(frame)
        classifications = (
        profile.loc[profile["source_csv"] == path.name]
            .set_index("measurement_column")["classification"]
            .to_dict()
        )
        columns = [c for c in columns if c in classifications]
        dt = parse_datetime(frame)
        order = dt.argsort()
        frame_sorted = frame.iloc[order].reset_index(drop=True)

        meta = per_column[per_column["source_csv"] == path.name].copy()
        unit_meta = unit_level[unit_level["source_csv"] == path.name].copy()

        top_cols = (
            meta.groupby("measurement_column")["anomaly_flag"]
            .mean()
            .sort_values(ascending=False)
            .head(args.top_n_columns)
            .index
        )

        plot_dir = args.output_dir / "plots" / station
        plot_dir.mkdir(parents=True, exist_ok=True)

        # EWMA charts for top-N columns
        for col in columns:
            values = pd.to_numeric(frame_sorted[col], errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(values)
            if valid.sum() < 20:
                continue
            cleaned = values[valid]
            ewma_result = ewma_control_chart(cleaned, tuple(args.ewma_lambdas), args.sigma_multiplier)
            lam_best = args.ewma_lambdas[0]
            key = f"ewma_{lam_best}"

            fig, ax = plt.subplots(figsize=(12, 5))
            indices = np.arange(len(cleaned))
            ax.plot(indices, cleaned, alpha=0.4, color="gray", linewidth=0.8, label="Raw")
            ax.plot(indices, ewma_result[key], color="steelblue", linewidth=1.5, label=f"EWMA (λ={lam_best})")
            ax.axhline(ewma_result[f"ucl_{lam_best}"][-1], color="red", linestyle="--", label="UCL")
            ax.axhline(ewma_result[f"lcl_{lam_best}"][-1], color="red", linestyle="--", label="LCL")
            # Flag out-of-control points
            ooc = (ewma_result[key] > ewma_result[f"ucl_{lam_best}"]) | (
                ewma_result[key] < ewma_result[f"lcl_{lam_best}"]
            )
            if ooc.any():
                ax.scatter(indices[ooc], ewma_result[key][ooc], color="red", s=30, label=f"OOC ({ooc.sum()})")
            ax.set(title=f"{station}: {col}  (last EWMA {ewma_result[key][-1]:.3f})",
                   xlabel="Unit sequence", ylabel="Value")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{safe_name(col)}_ewma.png", dpi=150)
            plt.close(fig)

        # Rolling anomaly rate trend
        if not unit_meta.empty:
            unit_meta_sorted = unit_meta.iloc[order].reset_index(drop=True)
            roll_rate = rolling_anomaly_rate(
                unit_meta_sorted["composite_anomaly_flag"].astype(float),
                window=args.anomaly_rate_window,
            )
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(roll_rate.index, roll_rate, color="darkorange", linewidth=1.5)
            ax.axhline(roll_rate.mean(), color="gray", linestyle="--", label="Mean rate")
            ax.set(title=f"{station}: Rolling composite anomaly rate (window={args.anomaly_rate_window})",
                   xlabel="Unit sequence (sorted by datetime)", ylabel="Anomaly rate")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / "rolling_anomaly_rate.png", dpi=150)
            plt.close(fig)

            # Trend: compare first half vs second half rates
            mid = len(unit_meta_sorted) // 2
            early_rate = unit_meta_sorted["composite_anomaly_flag"].iloc[:mid].mean()
            late_rate = unit_meta_sorted["composite_anomaly_flag"].iloc[mid:].mean()
            direction = "rising" if late_rate > early_rate * 1.2 else (
                "falling" if late_rate < early_rate * 0.8 else "stable")
        else:
            early_rate = late_rate = direction = np.nan

        # Per-column drift summary
        for col in columns:
            values = pd.to_numeric(frame_sorted[col], errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(values)
            if valid.sum() < 20:
                continue
            cleaned = values[valid]
            mid_idx = len(cleaned) // 2
            early_mean = cleaned[:mid_idx].mean()
            late_mean = cleaned[mid_idx:].mean()
            drift = late_mean - early_mean
            drift_pct = (drift / early_mean * 100) if early_mean != 0 else np.nan

            ewma_result = ewma_control_chart(cleaned, tuple(args.ewma_lambdas), args.sigma_multiplier)
            lam_best = args.ewma_lambdas[0]
            ooc_count = int(
                ((ewma_result[f"ewma_{lam_best}"] > ewma_result[f"ucl_{lam_best}"]) |
                 (ewma_result[f"ewma_{lam_best}"] < ewma_result[f"lcl_{lam_best}"]))
                .sum()
            )

            col_anomaly = meta[meta["measurement_column"] == col]
            col_anomaly_rate = col_anomaly["anomaly_flag"].mean() if not col_anomaly.empty else np.nan

            all_summaries.append({
                "station": station,
                "measurement_column": col,
                "n": int(valid.sum()),
                "overall_mean": float(cleaned.mean()),
                "overall_std": float(cleaned.std(ddof=1)),
                "early_half_mean": float(early_mean),
                "late_half_mean": float(late_mean),
                "drift": float(drift),
                "drift_pct": float(drift_pct) if not np.isnan(drift_pct) else None,
                "ewma_ooc_count": ooc_count,
                "ewma_ooc_rate": ooc_count / len(cleaned) if len(cleaned) else 0.0,
                "per_column_anomaly_rate": float(col_anomaly_rate) if not np.isnan(col_anomaly_rate) else None,
                "drift_direction": "rising" if drift_pct > 5 else ("falling" if drift_pct < -5 else "stable"),
            })

        print(f"  {station}: {len(columns)} columns, {len(frame_sorted)} units, "
              f"anomaly-rate trend: {direction} ({early_rate:.4f} → {late_rate:.4f})")

    summary = pd.DataFrame(all_summaries)
    summary.to_csv(args.output_dir / "forecast_trend_summary.csv", index=False)

    # Write a short narrative text
    rising = summary[summary["drift_direction"] == "rising"]
    falling = summary[summary["drift_direction"] == "falling"]
    lines = [
        "=== Forecast / Trend Summary ===",
        f"Analysed {len(args.inputs)} station files.",
        f"Columns with rising drift (>5% mean shift): {len(rising)}",
        f"Columns with falling drift (>5% mean shift): {len(falling)}",
        f"Columns with stable mean: {len(summary) - len(rising) - len(falling)}",
        "",
        "Columns with notable upward drift:",
    ]
    if not rising.empty:
        for _, row in rising.nlargest(10, "drift_pct").iterrows():
            lines.append(
                f"  {row['station']}/{row['measurement_column']}: "
                f"{row['drift_pct']:.1f}% drift, "
                f"{row['ewma_ooc_count']} EWMA OOC points"
            )
    else:
        lines.append("  (none)")
    lines.append(f"\nAnomaly-rate trends should be inspected in the rolling plots under {args.output_dir / 'plots'}/")
    (args.output_dir / "forecast_narrative.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(summary):,} column trend summaries → {args.output_dir / 'forecast_trend_summary.csv'}")
    print(f"Wrote narrative → {args.output_dir / 'forecast_narrative.txt'}")


if __name__ == "__main__":
    main()