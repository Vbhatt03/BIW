#!/usr/bin/env python3
"""Filter three station CSVs to their shared JSNs and build a merged table.

By default this writes exactly three CSVs, one filtered version of each input:
``data_stn08_common_jsn.csv``, ``data_stn18_common_jsn.csv``, and
``data_biwpc_common_jsn.csv``.  The merged table is constructed in memory;
pass ``--merged-output`` if it should also be written to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STATIONS = ("stn08", "stn18", "biw")


def read_station_csv(path: Path) -> pd.DataFrame:
    """Read a station file and normalise JSN without changing measurement data."""
    frame = pd.read_csv(path, dtype={"JSN": "string"})
    # These source files were exported with a pandas index.  It is not station
    # data and retaining it only creates an unnecessary merge column.
    frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed: \\d+$")].copy()
    if "JSN" not in frame.columns:
        raise ValueError(f"{path} has no JSN column")

    frame["JSN"] = frame["JSN"].str.strip().replace("", pd.NA)
    if frame["JSN"].isna().any():
        raise ValueError(f"{path} contains blank JSN values")
    return frame


def station_view_for_merge(frame: pd.DataFrame, station: str) -> pd.DataFrame:
    """Make a one-row-per-JSN, station-suffixed view for the working merge.

    Repeat measurements are retained in the station output CSVs, but a merge
    needs one row per unit.  Keeping the last file-order occurrence gives a
    deterministic result and normally selects the most recent remeasurement.
    """
    deduplicated = frame.drop_duplicates(subset="JSN", keep="last")
    renamed = deduplicated.rename(
        columns={column: f"{column}_{station}" for column in deduplicated.columns if column != "JSN"}
    )
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stn08", type=Path, default=Path("data_stn08.csv"))
    parser.add_argument("--stn18", type=Path, default=Path("data_stn18.csv"))
    parser.add_argument("--biw", type=Path, default=Path("data_biwpc.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("common_jsn_output"))
    parser.add_argument(
        "--merged-output",
        type=Path,
        help="Optional path for the station-suffixed merged working table.",
    )
    args = parser.parse_args()

    paths = {"stn08": args.stn08, "stn18": args.stn18, "biw": args.biw}
    frames = {station: read_station_csv(path) for station, path in paths.items()}
    jsn_sets = {station: set(frame["JSN"]) for station, frame in frames.items()}
    common = set.intersection(*(jsn_sets[station] for station in STATIONS))

    print("Unique-JSN overlap report")
    print(f"  |S08|     = {len(jsn_sets['stn08']):,}")
    print(f"  |S18|     = {len(jsn_sets['stn18']):,}")
    print(f"  |S_BIW|   = {len(jsn_sets['biw']):,}")
    print(f"  |common|  = {len(common):,}")
    for station in STATIONS:
        total = len(jsn_sets[station])
        dropped = len(jsn_sets[station] - common)
        percentage = (100 * dropped / total) if total else 0.0
        print(f"  {station}: dropped {dropped:,}/{total:,} ({percentage:.2f}%)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered: dict[str, pd.DataFrame] = {}
    for station in STATIONS:
        # Preserve source order and any repeated measurements in the three
        # requested station-specific outputs.
        filtered[station] = frames[station].loc[frames[station]["JSN"].isin(common)].copy()
        output_path = args.output_dir / f"{paths[station].stem}_common_jsn.csv"
        filtered[station].to_csv(output_path, index=False)
        print(f"Wrote {len(filtered[station]):,} rows: {output_path}")

    merged = station_view_for_merge(filtered["stn08"], "stn08")
    for station in ("stn18", "biw"):
        merged = merged.merge(
            station_view_for_merge(filtered[station], station), on="JSN", how="inner", validate="one_to_one"
        )
    if len(merged) != len(common):
        raise RuntimeError(f"Merged table has {len(merged)} rows; expected {len(common)} common JSNs")
    print(f"Merged working table: {len(merged):,} units and {len(merged.columns):,} columns")

    if args.merged_output:
        args.merged_output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.merged_output, index=False)
        print(f"Wrote merged working table: {args.merged_output}")


if __name__ == "__main__":
    main()
