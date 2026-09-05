#!/usr/bin/env python3
"""Detect per-measurement and cross-measurement anomalies for station CSVs.

The analysis never pools measurements across part columns or stations.  It
uses the classifications in ``distribution_profile_all_columns.csv`` to select
the univariate method: modified Z-score for Normal/Approximately Normal and
IQR fences for Non-normal columns.  Cross-part Isolation Forest and PCA are
fit separately for each input CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUTS = (Path("data_stn08.csv"), Path("data_stn18.csv"), Path("data_biwpc.csv"))
METADATA_COLUMNS = {"JSN", "DATE", "TIME", "Lab#", "File Name"}


def measurement_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name for name in frame.columns
        if name not in METADATA_COLUMNS and not name.startswith("Unnamed:")
    ]


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return percentile ranks in [0, 1], without requiring scipy."""
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def column_anomalies(
    values: pd.Series,
    classification: str,
    modified_z_threshold: float,
    iqr_multiplier: float,
) -> pd.DataFrame:
    """Return a score and flag for one part column; missing values are unscored."""
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.DataFrame(index=values.index)
    output["measurement_value"] = numeric
    output["anomaly_score"] = np.nan
    output["anomaly_flag"] = False

    observed = numeric.dropna()
    if observed.empty:
        output["method"] = "No numeric data"
        return output

    if classification in {"Normal", "Approximately Normal"}:
        median = observed.median()
        mad = (observed - median).abs().median()
        if mad > 0:
            # 0.6745 makes MAD comparable to standard deviation under normality.
            signed_score = 0.6745 * (numeric - median) / mad
            output["anomaly_score"] = signed_score.abs()
            output["anomaly_flag"] = output["anomaly_score"] > modified_z_threshold
            output["method"] = "modified_z_score"
        else:
            # A constant / nearly constant part cannot have a meaningful MAD.
            output["method"] = "modified_z_score_unavailable_zero_MAD"
    else:
        q1, q3 = observed.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            lower_fence = q1 - iqr_multiplier * iqr
            upper_fence = q3 + iqr_multiplier * iqr
            # Score is the excess beyond the nearest fence, expressed in IQRs.
            output["anomaly_score"] = np.maximum(
                (lower_fence - numeric) / iqr,
                (numeric - upper_fence) / iqr,
            ).clip(lower=0)
            output["anomaly_flag"] = output["anomaly_score"] > 0
            output["method"] = f"iqr_fence_{iqr_multiplier:g}x"
        else:
            output["method"] = "iqr_fence_unavailable_zero_IQR"
    return output


def cross_part_anomalies(
    measurements: pd.DataFrame,
    contamination: float,
    pca_variance: float,
    pca_percentile: float,
    random_state: int,
) -> pd.DataFrame:
    """Fit Isolation Forest and PCA reconstruction error within one station."""
    numeric = measurements.apply(pd.to_numeric, errors="coerce")
    usable = [name for name in numeric if numeric[name].notna().sum() >= 2 and numeric[name].nunique(dropna=True) > 1]
    result = pd.DataFrame(index=measurements.index)
    if len(usable) < 2 or len(measurements) < 10:
        result["iforest_anomaly_score"] = np.nan
        result["iforest_anomaly_flag"] = False
        result["pca_reconstruction_error"] = np.nan
        result["pca_anomaly_flag"] = False
        result["composite_anomaly_score"] = np.nan
        result["composite_anomaly_flag"] = False
        result["cross_part_note"] = "Not enough varying columns/rows for cross-part model"
        return result

    scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(numeric[usable]))
    forest = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    forest.fit(scaled)
    # Negating decision_function gives a directionally intuitive score: larger
    # values are more anomalous.  The IF prediction supplies the threshold flag.
    iforest_score = -forest.decision_function(scaled)
    iforest_flag = forest.predict(scaled) == -1

    pca = PCA(n_components=pca_variance, svd_solver="full", random_state=random_state)
    projected = pca.fit_transform(scaled)
    reconstructed = pca.inverse_transform(projected)
    reconstruction_error = np.mean((scaled - reconstructed) ** 2, axis=1)
    pca_cutoff = np.quantile(reconstruction_error, pca_percentile)
    pca_flag = reconstruction_error > pca_cutoff

    result["iforest_anomaly_score"] = iforest_score
    result["iforest_anomaly_flag"] = iforest_flag
    result["pca_reconstruction_error"] = reconstruction_error
    result["pca_anomaly_flag"] = pca_flag
    result["composite_anomaly_score"] = (percentile_rank(iforest_score) + percentile_rank(reconstruction_error)) / 2
    result["composite_anomaly_flag"] = iforest_flag | pca_flag
    result["cross_part_note"] = f"PCA retains {pca.explained_variance_ratio_.sum():.1%} variance; error flag > p{pca_percentile * 100:g}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--profile", type=Path, default=Path("distribution_profiles/distribution_profile_all_columns.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("anomaly_analysis"))
    parser.add_argument("--modified-z-threshold", type=float, default=3.5)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--contamination", type=float, default=0.02, help="Expected cross-part IF anomaly fraction.")
    parser.add_argument("--pca-variance", type=float, default=0.95, help="Fraction of scaled variance retained by PCA.")
    parser.add_argument("--pca-percentile", type=float, default=0.99, help="Reconstruction-error flag percentile.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.profile.exists():
        parser.error(f"Distribution profile not found: {args.profile}. Run profile_measurement_distributions.py first.")
    if not 0 < args.contamination < 0.5 or not 0 < args.pca_variance <= 1 or not 0 < args.pca_percentile < 1:
        parser.error("contamination, pca-variance, and pca-percentile must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = pd.read_csv(args.profile)
    all_column_rows: list[pd.DataFrame] = []
    all_unit_rows: list[pd.DataFrame] = []

    for path in args.inputs:
        frame = pd.read_csv(path)
        station = path.stem
        columns = measurement_columns(frame)
        classifications = profile.loc[profile["source_csv"] == path.name].set_index("measurement_column")["classification"].to_dict()
        unknown = [column for column in columns if column not in classifications]
        if unknown:
            raise ValueError(f"{path}: {len(unknown)} columns are absent from {args.profile}; regenerate the distribution profile first.")

        unit_id = pd.DataFrame({
            "source_csv": path.name,
            "station": station,
            "source_row": frame.index,
            "JSN": frame["JSN"].astype("string"),
        })
        for column in columns:
            per_column = column_anomalies(
                frame[column], classifications[column], args.modified_z_threshold, args.iqr_multiplier
            )
            per_column = pd.concat([unit_id, per_column], axis=1)
            per_column["measurement_column"] = column
            per_column["distribution_classification"] = classifications[column]
            all_column_rows.append(per_column)

        cross_part = cross_part_anomalies(
            frame[columns], args.contamination, args.pca_variance, args.pca_percentile, args.random_state
        )
        all_unit_rows.append(pd.concat([unit_id, cross_part], axis=1))
        print(f"Processed {path}: {len(frame):,} units, {len(columns):,} measurement columns")

    unit_column = pd.concat(all_column_rows, ignore_index=True)
    unit_level = pd.concat(all_unit_rows, ignore_index=True)
    column_summary = (
        unit_column.groupby(["source_csv", "station", "measurement_column", "distribution_classification", "method"], dropna=False)
        .agg(units_with_values=("measurement_value", "count"), anomaly_count=("anomaly_flag", "sum"), anomaly_rate=("anomaly_flag", "mean"))
        .reset_index()
    )
    station_summary = (
        unit_level.groupby(["source_csv", "station"], dropna=False)
        .agg(units=("JSN", "size"), iforest_anomaly_rate=("iforest_anomaly_flag", "mean"), pca_anomaly_rate=("pca_anomaly_flag", "mean"), composite_anomaly_rate=("composite_anomaly_flag", "mean"))
        .reset_index()
    )

    unit_column.to_csv(args.output_dir / "unit_by_column_anomalies.csv", index=False)
    unit_level.to_csv(args.output_dir / "unit_level_composite_anomalies.csv", index=False)
    column_summary.to_csv(args.output_dir / "anomaly_rates_by_column_and_station.csv", index=False)
    station_summary.to_csv(args.output_dir / "anomaly_rates_by_station.csv", index=False)
    print(f"Wrote anomaly outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
