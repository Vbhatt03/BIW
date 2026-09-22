#!/usr/bin/env python3
"""Root-cause analysis demo using DoWhy with side-grouped anomaly flags.

Groups measurements into Left / Right / Center per station (from columns.txt
annotations), then tests causal questions like:

  "Does a left-side anomaly at STN08 cause a left-side anomaly at STN18?"

WARNING: Demonstration framework only — no validated root-cause labels exist.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import pandas as pd
from dowhy import CausalModel

DEFAULT_ANOMALY_DIR = Path("anomaly_analysis")
DEFAULT_OUTPUT_DIR = Path("rca_output")

SIDE_GROUPS: dict[str, dict[str, list[str]]] = {
    "stn08": {
        "L": ["H9451DA01_X", "H9451DA01_Y", "H9451DA01_Z", "L9393MA01_Z"],
        "R": ["L9451DA02_X", "L9451DA02_Z", "L9392MA01_Z"],
    },
    "stn18": {
        "L": ["H9448DA02_X", "H9448DA02_Y", "F9393A792_Y", "F9393A792_Z", "G9393A792_Y"],
        "R": ["L9448DA01_X", "F9392A792_Y", "F9392A792_Z", "G9392A792_Y"],
    },
    "biwpc": {
        "L": ["FB33A033_D", "FB23A797_D", "GB22A797_D", "GB32A033_D"],
        "C": ["FB33A001_D", "GB32A001_D"],
        "R": ["FB33A037R_D", "GB32A0037R_D", "GB22A796R_D"],
    },
}

# Causal questions: (upstream_station, upstream_side, downstream_station, downstream_side)
CAUSAL_PAIRS: list[tuple[str, str, str, str]] = [
    ("stn08", "L", "stn18", "L"),
    ("stn08", "R", "stn18", "R"),
    ("stn18", "L", "biwpc", "L"),
    ("stn18", "R", "biwpc", "R"),
    ("stn08", "L", "biwpc", "L"),
    ("stn08", "R", "biwpc", "R"),
    ("stn18", "L", "biwpc", "C"),
    ("stn18", "R", "biwpc", "C"),
]


def side_anomaly_flag(row: pd.Series, station: str, side: str) -> bool:
    """True if ANY column in this station/side group is anomalous."""
    for col in SIDE_GROUPS[station].get(side, []):
        val = row.get(f"{station}_{col}_anomaly", 0)
        if val == 1 or val is True:
            return True
    return False


def load_and_group_anomalies(anomaly_dir: Path) -> pd.DataFrame:
    anom = pd.read_csv(anomaly_dir / "unit_by_column_anomalies.csv", dtype={"JSN": "string"})

    jsns = anom["JSN"].unique()
    rows: list[dict] = []
    for jsn in jsns:
        sub = anom[anom["JSN"] == jsn]
        row: dict = {"JSN": jsn}
        for _, r in sub.iterrows():
            key = f"{r['station']}_{r['measurement_column']}_anomaly"
            row[f"{r['station'].removeprefix('data_')}_{r['measurement_column']}_anomaly"] = int(bool(r["anomaly_flag"]))
        rows.append(row)

    wide = pd.DataFrame(rows).fillna(0)

    for station, sides in SIDE_GROUPS.items():
        for side in sides:
            key = f"{station}_side_{side}_anomaly"
            cols = [f"{station}_{c}_anomaly" for c in SIDE_GROUPS[station][side]
                    if f"{station}_{c}_anomaly" in wide.columns]
            if cols:
                wide[key] = wide[cols].any(axis=1).astype(int)
            else:
                wide[key] = 0

    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomaly-dir", type=Path, default=DEFAULT_ANOMALY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading anomaly data and building side-grouped flags...")
    data = load_and_group_anomalies(args.anomaly_dir)
    print(f"Units with side-grouped data: {len(data)}")

    results: list[dict] = []
    narrative = ["=== DoWhy Causal Demo: Side-Grouped Anomalies ===\n"]

    for up_stn, up_side, down_stn, down_side in CAUSAL_PAIRS:
        treat = f"{up_stn}_side_{up_side}_anomaly"
        outcome = f"{down_stn}_side_{down_side}_anomaly"

        if treat not in data.columns or outcome not in data.columns:
            narrative.append(f"Skipping {up_stn} {up_side} → {down_stn} {down_side}: columns missing")
            continue

        sub = data[[treat, outcome]].dropna()
        sub[treat] = sub[treat].astype(int)
        sub[outcome] = sub[outcome].astype(int)

        if sub[treat].nunique() < 2 or sub[outcome].nunique() < 2:
            msg = f"Skipping {up_stn} {up_side} → {down_stn} {down_side}: no variance (treat={sub[treat].nunique()}, outcome={sub[outcome].nunique()})"
            print(msg)
            narrative.append(msg)
            continue

        print(f"\n{up_stn} {up_side} → {down_stn} {down_side}")
        print(f"  n={len(sub)}, treat rate={sub[treat].mean():.4f}, outcome rate={sub[outcome].mean():.4f}")

        p_t = sub.loc[sub[treat] == 1, outcome].mean() if sub[treat].sum() > 0 else 0
        p_c = sub.loc[sub[treat] == 0, outcome].mean() if (sub[treat] == 0).sum() > 0 else 0
        print(f"  P(outcome | treat=1) = {p_t:.4f}")
        print(f"  P(outcome | treat=0) = {p_c:.4f}")

        try:
            import networkx as nx
            sub_conf = sub.assign(unit_index=np.arange(len(sub)))
            model = CausalModel(
                data=sub_conf,
                treatment=treat,
                outcome=outcome,
                graph=nx.DiGraph([
                    ("unit_index", treat),
                    ("unit_index", outcome),
                    (treat, outcome),
                ]),
            )
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified, method_name="backdoor.propensity_score_matching"
            )
            ref_p = model.refute_estimate(identified, estimate, "placebo_treatment_refuter", placebo_type="permute")
            ref_c = model.refute_estimate(identified, estimate, "random_common_cause")
            ate = estimate.value
            print(f"  ATE={ate:.4f}, placebo→{ref_p.new_effect:.4f}, cc→{ref_c.new_effect:.4f}")

            results.append({
                "question": f"{up_stn} {up_side} → {down_stn} {down_side}",
                "n": len(sub),
                "treat_rate": sub[treat].mean(),
                "outcome_rate": sub[outcome].mean(),
                "cond_risk_treat": round(p_t, 4),
                "cond_risk_control": round(p_c, 4),
                "ate": round(ate, 4),
                "placebo_ate": round(ref_p.new_effect, 4),
                "cc_ate": round(ref_c.new_effect, 4),
            })
        except Exception as e:
            print(f"  DoWhy failed: {e}")
            narrative.append(f"{up_stn} {up_side} → {down_stn} {down_side}: failed ({e})")
            continue

    pd.DataFrame(results).to_csv(args.output_dir / "rca_dowhy_results.csv", index=False)
    print(f"\nWrote {len(results)} results → {args.output_dir / 'rca_dowhy_results.csv'}")
    narrative.append(f"\nQuestions tested: {len(results)}")
    for r in results:
        narrative.append(f"  {r['question']}: ATE={r['ate']}")
    narrative.append("\nDISCLAIMER: Demonstration only — no validated root-cause labels.")
    (args.output_dir / "rca_narrative.txt").write_text("\n".join(narrative))

    print("Done.")


if __name__ == "__main__":
    main()