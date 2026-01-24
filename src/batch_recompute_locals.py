#!/usr/bin/env python3
"""
Batch Recompute Local Eigenvalues Script

Optimized script that only recomputes local eigenvalues using the new bipartite
matching algorithm. Skips recomputing:
- Equitable Partition (loaded from pickle)
- Local Equitable Partitions (loaded from pickle)
- Divisor matrix global eigenvalues (unchanged)

Only recomputes:
- Local eigenvalues (affected by getSetDifference matching change)

Usage:
    python batch_recompute_locals.py [--min-threshold 5] [--threshold 60] [--dry-run]
"""

import argparse
import pickle
import sys
from pathlib import Path
from time import time
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import sparse

import ep_utils
import ep_finder
import lep_finder


# Configuration
METRIC_WARDEN_PATH = Path(__file__).parent / "MetricWarden.csv"
DT_STATS_DIR = Path(__file__).parent.parent / "DT_Stats"
RESULTS_DIR = Path(__file__).parent.parent / "Results"
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
    df.columns = df.columns.str.strip()
    return df


def load_partition_data(name: str) -> tuple:
    """Load EP and LEP data from pickle files. Returns (None, None) if missing or empty."""
    results_dir = RESULTS_DIR / name

    ep_file = results_dir / "ep_data.pkl"
    lep_file = results_dir / "lep_data.pkl"

    if not ep_file.exists() or not lep_file.exists():
        return None, None

    try:
        with open(ep_file, 'rb') as f:
            pi = pickle.load(f)
        with open(lep_file, 'rb') as f:
            leps = pickle.load(f)

        # Check if data is valid (not empty)
        if not pi or not leps:
            return None, None

        return pi, leps
    except Exception as e:
        print(f"    Warning: Error loading partition data: {e}")
        return None, None


def compute_partition_data(csr: sparse.csr_array) -> tuple:
    """Compute EP and LEP data from scratch."""
    csc = csr.tocsc()

    # Compute coarsest equitable partition
    pi = ep_finder.getEquitablePartition(ep_finder.initFromSparse(csr))

    # Compute local equitable partitions
    leps = lep_finder.getLocalEquitablePartitions(lep_finder.initFromSparse(csc), pi)

    return pi, leps


def save_partition_data(name: str, pi: Dict[int, List[Any]], leps: List[List[int]]) -> None:
    """Save EP and LEP data to pickle files."""
    results_dir = RESULTS_DIR / name
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "ep_data.pkl", 'wb') as f:
        pickle.dump(pi, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(results_dir / "lep_data.pkl", 'wb') as f:
        pickle.dump(leps, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(source_file: str, directed: bool) -> sparse.csr_array:
    """Load graph from source file."""
    import graphs
    mat = graphs.oneGraphToRuleThemAll(source_file, visualize=False, directed=directed, suppress=True)
    return mat.tocsr()


def recompute_locals_only(
    csr: sparse.csr_array,
    pi: Dict[int, List[int]],
    leps: List[List[int]],
) -> List[float | complex]:
    """
    Recompute only the local eigenvalues using the new bipartite matching.

    This is equivalent to ep_utils.getLocals() but we compute the divisor matrix
    here since it's not stored.
    """
    # Compute divisor matrix (needed for subgraph divisor submatrices)
    csc = csr.tocsc()
    divisor_matrix = ep_utils.getDivisorMatrixSparse(csc, pi)

    # Now compute locals (this uses the new getSetDifference with bipartite matching)
    locals_eigs = ep_utils.getLocals(csr, divisor_matrix, pi, leps)

    return locals_eigs


def save_locals(name: str, locals_eigs: List[float | complex]) -> None:
    """Save local eigenvalues to DT_Stats."""
    stats_dir = DT_STATS_DIR / f"{name}_stats"

    with open(stats_dir / "locals.txt", 'w') as f:
        for local_val in locals_eigs:
            f.write(f"{local_val}\n")


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


def process_dataset(name: str, source_file: str, directed: bool, dry_run: bool = False) -> bool:
    """Process a single dataset - recompute only locals (or full EP/LEP if missing)."""

    # Load partition data
    pi, leps = load_partition_data(name)
    need_partition_compute = (pi is None)

    if need_partition_compute:
        print(f"    Partition data missing/empty - will compute EP and LEPs")

    if dry_run:
        if need_partition_compute:
            print(f"    Would compute EP and LEPs, then locals")
        else:
            print(f"    Would recompute locals for {len(leps)} LEPs")
        return True

    # Load graph
    try:
        csr = load_graph(source_file, directed)
    except Exception as e:
        print(f"    Error loading graph: {e}")
        return False

    # Compute partition data if missing
    if need_partition_compute:
        try:
            print(f"    Computing EP and LEPs...")
            pi, leps = compute_partition_data(csr)
            save_partition_data(name, pi, leps)
            print(f"    Saved partition data ({len(pi)} EP elements, {len(leps)} LEPs)")
        except Exception as e:
            print(f"    Error computing partition data: {e}")
            return False

    # Recompute only locals
    try:
        locals_eigs = recompute_locals_only(csr, pi, leps)
    except Exception as e:
        print(f"    Error computing locals: {e}")
        return False

    # Save new locals
    save_locals(name, locals_eigs)
    print(f"    Saved {len(locals_eigs)} local eigenvalues")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch recompute local eigenvalues only (optimized for matching change)"
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
    print(f"Loaded {len(df)} entries from MetricWarden.csv")

    # Get column names
    time_col = [c for c in df.columns if "m_total_time" in c][0]
    source_col = [c for c in df.columns if "m_source_file" in c][0]
    directed_col = [c for c in df.columns if "g_directed" in c][0]

    # Filter by threshold and deduplicate by name (keep first occurrence)
    qualifying = df[(df[time_col] >= args.min_threshold) & (df[time_col] < args.threshold)].copy()
    qualifying = qualifying.drop_duplicates(subset=['name'], keep='first')
    print(f"Found {len(qualifying)} unique datasets with {args.min_threshold}s <= m_total_time < {args.threshold}s\n")

    if len(qualifying) == 0:
        print("No datasets to process.")
        return

    # Sort by time (fastest first)
    qualifying = qualifying.sort_values(time_col)

    # Estimate total time (locals computation is ~10-20% of total)
    estimated_total = qualifying[time_col].sum() * 0.15
    print(f"Estimated total time: {format_time(estimated_total)} (locals only)")
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

        print(f"\n[{i}/{len(qualifying)}] {name} (prev total: {format_time(prev_time)})")

        # Check if source file exists
        if not Path(source_file).exists():
            print(f"    Skipping: source file not found")
            skipped += 1
            continue

        # Remove old bipartite files
        remove_bipartite_files(name, args.dry_run)

        # Process dataset
        dataset_start = time()
        success = process_dataset(name, source_file, directed, args.dry_run)
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
            print(f"    Progress: {elapsed:.0f}s elapsed, ~{format_time(estimated_remaining)} remaining")

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
