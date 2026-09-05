#!/usr/bin/env python3
"""Profile each measurement column separately for the three station CSVs.

No measurements are pooled: every numeric measurement column is analysed within
its own input file.  The script writes one combined profile table and a
histogram/KDE plus Q-Q plot for each analysed column.

Spec limits (LSL/USL) are read from the corresponding meta_*.csv files (row 7 =
LSL, row 8 = USL).  When present, the histogram bars whose centre falls outside
the spec range are coloured red, vertical red lines mark the limits, and a
green dashed KDE is drawn for the within-spec data.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks

COLUMNS_CONFIG_PATH = Path("columns.txt")


def _allowed_columns(station: str) -> set[str]:
    key = station.removeprefix("data_")
    namespace: dict = {}
    exec(COLUMNS_CONFIG_PATH.read_text(), namespace)
    columns: set[str] = set()
    suffix = f"_{key}"
    for var_name, value in namespace.items():
        if var_name.startswith("colnames") and isinstance(value, list) and var_name.endswith(suffix):
            columns.update(value)
    return columns


DEFAULT_INPUTS = (Path("data_stn08.csv"), Path("data_stn18.csv"), Path("data_biwpc.csv"))
NON_MEASUREMENT_COLUMNS = {"JSN", "DATE", "TIME", "Lab#", "File Name"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "column"


def measurement_columns(frame: pd.DataFrame, station: str) -> list[str]:
    allowed = _allowed_columns(station)
    return [
        name for name in frame.columns
        if name in allowed and name not in NON_MEASUREMENT_COLUMNS and not name.startswith("Unnamed:")
        and pd.to_numeric(frame[name], errors="coerce").notna().any()
    ]


def _load_spec_limits(station: str) -> dict[str, tuple[float | None, float | None]]:
    key = station.removeprefix("data_")
    meta_path = Path(f"meta_{key}.csv")
    if not meta_path.exists():
        return {}
    meta = pd.read_csv(meta_path, header=0)
    col_names = [c for c in meta.columns if not c.startswith("Unnamed") and str(c).strip()]
    if len(meta) < 7:
        return {}
    lsl_row = meta.iloc[5]
    usl_row = meta.iloc[6]
    limits = {}
    for c in col_names:
        try:
            lsl = float(lsl_row[c]) if pd.notna(lsl_row[c]) else None
            usl = float(usl_row[c]) if pd.notna(usl_row[c]) else None
        except (ValueError, TypeError):
            lsl, usl = None, None
        limits[c] = (lsl, usl)
    return limits


def bimodal_peak_count(values: np.ndarray) -> int:
    if len(values) < 20 or np.ptp(values) == 0:
        return 0
    try:
        kde = stats.gaussian_kde(values)
        grid = np.linspace(values.min(), values.max(), 512)
        density = kde(grid)
        peaks, _ = find_peaks(density, prominence=max(density) * 0.08, distance=30)
        return len(peaks)
    except (np.linalg.LinAlgError, ValueError):
        return 0


def shape_and_recommendation(values: np.ndarray, skewness: float, kurtosis: float) -> tuple[str, str, int]:
    unique, counts = np.unique(values, return_counts=True)
    endpoint_share = max(counts[0], counts[-1]) / len(values) if len(unique) else 0.0
    peaks = bimodal_peak_count(values)
    labels: list[str] = []

    if peaks >= 2:
        labels.append("potentially bimodal/multimodal")
    if skewness >= 0.75:
        labels.append("right-skewed")
    elif skewness <= -0.75:
        labels.append("left-skewed")
    if kurtosis >= 1.5:
        labels.append("heavy-tailed")
    elif kurtosis <= -1.0:
        labels.append("light-tailed/flat")
    if endpoint_share >= 0.05:
        labels.append("possible truncation/rounding at an endpoint")
    if not labels:
        labels.append("approximately symmetric/unimodal")

    if peaks >= 2:
        recommendation = "Investigate/stratify possible subpopulations; do not transform mixtures as one population."
    elif endpoint_share >= 0.05:
        recommendation = "Investigate a possible spec-limit, rounding, or censoring artifact before transforming."
    elif abs(skewness) >= 0.75 and np.all(values > 0):
        recommendation = "Consider log or Box-Cox only if a normality-assuming downstream method is required."
    elif abs(skewness) >= 0.75:
        recommendation = "Consider Yeo-Johnson (supports zero/negative values) before normality-assuming methods."
    elif kurtosis >= 1.5:
        recommendation = "Prefer robust/non-parametric methods; inspect tails before using 3-sigma or Grubbs' limits."
    else:
        recommendation = "No transform indicated by this screen."
    return "; ".join(labels), recommendation, peaks


def classify_normality(
    shapiro_p: float,
    normaltest_p: float,
    skewness: float,
    kurtosis: float,
    peaks: int,
    alpha: float,
) -> str:
    valid_p_values = [p for p in (shapiro_p, normaltest_p) if not np.isnan(p)]
    visual_concerns = peaks >= 2 or abs(skewness) >= 0.75 or abs(kurtosis) >= 1.5
    if valid_p_values and all(p >= alpha for p in valid_p_values) and not visual_concerns:
        return "Normal"
    if valid_p_values and all(p >= alpha / 5 for p in valid_p_values) and not visual_concerns:
        return "Approximately Normal"
    return "Non-normal"


def save_plots(values: np.ndarray, station: str, column: str, classification: str, output_dir: Path,
               lsl: float | None = None, usl: float | None = None) -> tuple[Path, Path]:
    station_dir = output_dir / "plots" / safe_name(station)
    station_dir.mkdir(parents=True, exist_ok=True)
    base = safe_name(column)
    histogram_path = station_dir / f"{base}_hist_kde.png"
    qq_path = station_dir / f"{base}_qq.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    counts, bins = np.histogram(values, bins="auto", density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)
    bar_colors = []
    for bc in bin_centers:
        if lsl is not None and usl is not None and (bc < lsl or bc > usl):
            bar_colors.append("red")
        else:
            bar_colors.append("#3977af")
    ax.bar(bin_centers, counts, width=widths, color=bar_colors, alpha=0.7, align="center")

    if len(np.unique(values)) > 1:
        sns.kdeplot(x=values, ax=ax, color="blue", linewidth=2, label="KDE (all)")

    if lsl is not None and usl is not None:
        within = values[(values >= lsl) & (values <= usl)]
        if len(within) > 1 and len(np.unique(within)) > 1:
            sns.kdeplot(x=within, ax=ax, color="green", linewidth=2, linestyle="--", label="KDE (within spec)")

    if lsl is not None:
        ax.axvline(lsl, color="red", linestyle="--", linewidth=1.5, label=f"LSL={lsl:.3g}")
    if usl is not None:
        ax.axvline(usl, color="red", linestyle="--", linewidth=1.5, label=f"USL={usl:.3g}")

    ax.set(title=f"{station}: {column}\n{classification} (n={len(values):,})", xlabel=column, ylabel="Density")
    if lsl is not None or usl is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(f"{station}: {column} Q-Q plot")
    fig.tight_layout()
    fig.savefig(qq_path, dpi=150)
    plt.close(fig)
    return histogram_path, qq_path


def profile_column(values: np.ndarray, alpha: float) -> dict[str, object]:
    n = len(values)
    if n < 3 or np.ptp(values) == 0:
        return {
            "n": n, "mean": np.mean(values) if n else np.nan, "median": np.median(values) if n else np.nan,
            "std": np.std(values, ddof=1) if n > 1 else np.nan, "min": np.min(values) if n else np.nan,
            "max": np.max(values) if n else np.nan, "iqr": np.nan, "skewness": np.nan, "kurtosis": np.nan,
            "shapiro_statistic": np.nan, "shapiro_p_value": np.nan,
            "normaltest_statistic": np.nan, "normaltest_p_value": np.nan,
            "classification": "Non-normal", "likely_shape": "constant or too few observations",
            "kde_peak_count": 0, "recommendation": "No normality-based method; collect more variation/observations.",
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shapiro_stat, shapiro_p = stats.shapiro(values)
        if n >= 8:
            normaltest_stat, normaltest_p = stats.normaltest(values)
        else:
            normaltest_stat, normaltest_p = np.nan, np.nan
        skewness = stats.skew(values, bias=False)
        kurtosis = stats.kurtosis(values, fisher=True, bias=False)
    shape, recommendation, peaks = shape_and_recommendation(values, skewness, kurtosis)
    classification = classify_normality(shapiro_p, normaltest_p, skewness, kurtosis, peaks, alpha)
    return {
        "n": n, "mean": np.mean(values), "median": np.median(values), "std": np.std(values, ddof=1),
        "min": np.min(values), "max": np.max(values),
        "iqr": stats.iqr(values), "skewness": skewness, "kurtosis": kurtosis,
        "shapiro_statistic": shapiro_stat, "shapiro_p_value": shapiro_p,
        "normaltest_statistic": normaltest_stat, "normaltest_p_value": normaltest_p,
        "classification": classification, "likely_shape": shape, "kde_peak_count": peaks,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS), help="Station CSVs to analyse.")
    parser.add_argument("--output-dir", type=Path, default=Path("distribution_profiles"))
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Normality-test significance cutoff, in (0, 1); default: 0.05. Try 0.01 as a sensitivity analysis.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Optional profile-table path (useful when profiling files in separate runs).",
    )
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be strictly between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        station = path.stem
        columns = measurement_columns(frame, station)
        spec_limits = _load_spec_limits(station)
        print(f"Profiling {path}: {len(columns)} candidate measurement columns")
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
            result: dict[str, object] = {"source_csv": path.name, "station": station, "measurement_column": column}
            result.update(profile_column(values, args.alpha))
            result["alpha"] = args.alpha
            lsl, usl = spec_limits.get(column, (None, None))
            result["lsl"] = lsl
            result["usl"] = usl
            histogram_path, qq_path = save_plots(values, station, column, str(result["classification"]), args.output_dir, lsl, usl) if len(values) else (None, None)
            result["histogram_kde_plot"] = str(histogram_path) if histogram_path else ""
            result["qq_plot"] = str(qq_path) if qq_path else ""
            rows.append(result)

    profile = pd.DataFrame(rows)
    output_path = args.profile_output or args.output_dir / "distribution_profile_all_columns.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path, index=False)
    print(f"Wrote {len(profile):,} column profiles: {output_path}")
    print(profile["classification"].value_counts().to_string())


if __name__ == "__main__":
    main()