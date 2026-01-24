#!/usr/bin/env python3
"""
Batch Rerun LEParD Script

Reruns zookeeper.py in --lepard-only mode for datasets under a time threshold.
Also removes old bipartite matching files so they can be recomputed.

Usage:
    python batch_rerun_lepard.py [--min-threshold 5] [--threshold 60] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from time import time

import pandas as pd


# Configuration
METRIC_WARDEN_PATH = Path(__file__).parent / "MetricWarden.csv"
DT_STATS_DIR = Path(__file__).parent.parent / "DT_Stats"
ZOOKEEPER_PATH = Path(__file__).parent / "zookeeper.py"
DEFAULT_THRESHOLD = 60.0


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def load_metric_warden() -> pd.DataFrame:
    """Load MetricWarden.csv and parse relevant columns."""
    if not METRIC_WARDEN_PATH.exists():
        print(f"Error: MetricWarden.csv not found at {METRIC_WARDEN_PATH}")
        sys.exit(1)

    df = pd.read_csv(METRIC_WARDEN_PATH, skipinitialspace=True)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    return df


def remove_bipartite_files(dataset_name: str, dry_run: bool = False) -> None:
    """Remove matched_pairs_bipartite.csv and unmatched_bipartite.csv for a dataset."""
    stats_dir = DT_STATS_DIR / f"{dataset_name}_stats"

    files_to_remove = [
        stats_dir / "matched_pairs_bipartite.csv",
        stats_dir / "unmatched_bipartite.csv",
    ]

    for filepath in files_to_remove:
        if filepath.exists():
            if dry_run:
                print(f"    Would remove: {filepath.name}")
            else:
                filepath.unlink()
                print(f"    Removed: {filepath.name}")


def run_zookeeper(source_file: str, directed: bool, dry_run: bool = False) -> bool:
    """Run zookeeper.py --lepard-only on a dataset."""
    cmd = [sys.executable, str(ZOOKEEPER_PATH), "--lepard-only", "--file", source_file]
    if directed:
        cmd.append("--directed")

    if dry_run:
        print(f"    Would run: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode != 0:
            print(
                f"    Error: {result.stderr[:200] if result.stderr else 'Unknown error'}"
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    Timeout after 600s")
        return False
    except Exception as e:
        print(f"    Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch rerun LEParD for datasets under a time threshold"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Max m_total_time in seconds (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.0,
        help="Min m_total_time in seconds (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually running",
    )
    args = parser.parse_args()

    # Load data
    df = load_metric_warden()
    print(f"Loaded {len(df)} datasets from MetricWarden.csv")

    # Filter by threshold
    # Handle the column name which might have extra spaces
    time_col = [c for c in df.columns if "m_total_time" in c][0]
    source_col = [c for c in df.columns if "m_source_file" in c][0]
    directed_col = [c for c in df.columns if "g_directed" in c][0]

    qualifying = df[(df[time_col] >= args.min_threshold) & (df[time_col] < args.threshold)].copy()
    print(f"Found {len(qualifying)} datasets with {args.min_threshold}s <= m_total_time < {args.threshold}s\n")

    if len(qualifying) == 0:
        print("No datasets to process.")
        return

    # Sort by time (fastest first)
    qualifying = qualifying.sort_values(time_col)

    # Estimate total time
    estimated_total = qualifying[time_col].sum()
    print(f"Estimated total time: {format_time(estimated_total)}")
    print("=" * 60)

    # Process each dataset
    successful = 0
    failed = 0
    skipped = 0
    start_time = time()

    for i, row in enumerate(qualifying.itertuples(), 1):
        name = row.name
        source_file = getattr(row, source_col.replace(" ", "_").replace("-", "_"))
        directed = getattr(row, directed_col.replace(" ", "_").replace("-", "_"))
        prev_time = getattr(row, time_col.replace(" ", "_").replace("-", "_"))

        print(f"\n[{i}/{len(qualifying)}] {name} (prev: {format_time(prev_time)})")

        # Check if source file exists
        if not Path(source_file).exists():
            print(f"    Skipping: source file not found")
            skipped += 1
            continue

        # Remove old bipartite files
        remove_bipartite_files(name, args.dry_run)

        # Run zookeeper
        dataset_start = time()
        success = run_zookeeper(source_file, directed, args.dry_run)
        dataset_elapsed = time() - dataset_start

        if success:
            successful += 1
            print(f"    Completed in {format_time(dataset_elapsed)}")
        else:
            failed += 1

        # Progress update
        elapsed = time() - start_time
        remaining_datasets = len(qualifying) - i
        if i > 0:
            avg_time = elapsed / i
            estimated_remaining = avg_time * remaining_datasets
            print(
                f"    Progress: {elapsed:.0f}s elapsed, ~{format_time(estimated_remaining)} remaining"
            )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total datasets: {len(qualifying)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"Total time: {format_time(time() - start_time)}")


if __name__ == "__main__":
    main()
