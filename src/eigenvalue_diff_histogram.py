#!/usr/bin/env python3
"""
Eigenvalue Difference Histogram

Creates a histogram of all eigenvalue magnitude differences between NumPy and LEParD
across ALL datasets.

Usage:
    python eigenvalue_diff_histogram.py [--no-show]
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Configuration
DT_STATS_DIR = Path(__file__).parent.parent / "DT_Stats"
OUTPUT_DIR = Path(__file__).parent.parent / "eigenvalue_analysis"
ZERO_THRESHOLD = 1e-16  # For log scale plotting


def load_matched_pairs_bipartite(stats_dir: Path) -> Optional[pd.DataFrame]:
    """Load matched_pairs_bipartite.csv."""
    csv_path = stats_dir / "matched_pairs_bipartite.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df["diff_magnitude"] = df["diff_magnitude"].astype(float)
    return df


def get_all_datasets() -> List[str]:
    """Get list of all dataset names from DT_Stats directory."""
    if not DT_STATS_DIR.exists():
        return []

    datasets = []
    for item in DT_STATS_DIR.iterdir():
        if item.is_dir() and item.name.endswith("_stats"):
            if (item / "matched_pairs_bipartite.csv").exists():
                dataset_name = item.name[:-6]  # Remove "_stats" suffix
                datasets.append(dataset_name)

    return sorted(datasets)


def collect_all_differences() -> pd.DataFrame:
    """Collect all difference magnitudes across all datasets."""
    datasets = get_all_datasets()
    all_diffs = []

    print(f"Collecting differences from {len(datasets)} datasets...\n")

    for i, dataset_name in enumerate(datasets, 1):
        stats_dir = DT_STATS_DIR / f"{dataset_name}_stats"
        df = load_matched_pairs_bipartite(stats_dir)

        if df is None:
            continue

        df["dataset"] = dataset_name
        all_diffs.append(df[["diff_magnitude", "dataset"]])
        print(f"  [{i}/{len(datasets)}] {dataset_name}: {len(df)} pairs")

    if all_diffs:
        combined = pd.concat(all_diffs, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()


def create_histogram(df: pd.DataFrame, output_dir: Path, show_plots: bool = True) -> None:
    """Create histogram of all difference magnitudes."""
    if len(df) == 0:
        print("No data to plot.")
        return

    diff_mags = df["diff_magnitude"].values

    # Replace zeros with small value for log scale
    diff_mags_nonzero = diff_mags[diff_mags > 0]
    diff_mags_zero_count = np.sum(diff_mags == 0)

    print(f"\nStatistics:")
    print(f"  Total pairs: {len(diff_mags)}")
    print(f"  Exact matches (diff=0): {diff_mags_zero_count}")
    print(f"  Non-zero differences: {len(diff_mags_nonzero)}")
    if len(diff_mags_nonzero) > 0:
        print(f"  Min non-zero: {diff_mags_nonzero.min():.6e}")
        print(f"  Max: {diff_mags_nonzero.max():.6e}")
        print(f"  Mean: {diff_mags_nonzero.mean():.6e}")
        print(f"  Median: {np.median(diff_mags_nonzero):.6e}")

    # Create histogram with log scale bins
    fig, ax = plt.subplots(figsize=(12, 7))

    if len(diff_mags_nonzero) > 0:
        # Use log-spaced bins for non-zero values
        log_min = np.floor(np.log10(diff_mags_nonzero.min()))
        log_max = np.ceil(np.log10(diff_mags_nonzero.max()))
        bins = np.logspace(log_min, log_max, 50)

        ax.hist(diff_mags_nonzero, bins=bins, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xscale("log")

    ax.set_xlabel("|NumPy - LEParD| (magnitude of difference)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Eigenvalue Differences Across All Datasets\n"
                 f"({len(df)} total pairs, {diff_mags_zero_count} exact matches not shown)")
    ax.grid(True, alpha=0.3)

    # Add vertical lines for reference thresholds
    for thresh, color, label in [(1e-10, "green", "1e-10"), (1e-5, "orange", "1e-5"), (1e-2, "red", "1e-2")]:
        if len(diff_mags_nonzero) > 0 and diff_mags_nonzero.min() <= thresh <= diff_mags_nonzero.max():
            ax.axvline(x=thresh, color=color, linestyle="--", alpha=0.7, label=label)

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "all_differences_histogram.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # Also create a cumulative distribution
    fig, ax = plt.subplots(figsize=(12, 7))

    if len(diff_mags_nonzero) > 0:
        sorted_diffs = np.sort(diff_mags_nonzero)
        cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)

        ax.plot(sorted_diffs, cumulative, color="steelblue", linewidth=2)
        ax.set_xscale("log")

    ax.set_xlabel("|NumPy - LEParD| (magnitude of difference)")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title(f"Cumulative Distribution of Eigenvalue Differences\n"
                 f"({len(diff_mags_nonzero)} non-zero differences)")
    ax.grid(True, alpha=0.3)

    # Add reference lines
    for thresh, color, label in [(1e-10, "green", "1e-10"), (1e-5, "orange", "1e-5"), (1e-2, "red", "1e-2")]:
        if len(diff_mags_nonzero) > 0 and diff_mags_nonzero.min() <= thresh <= diff_mags_nonzero.max():
            ax.axvline(x=thresh, color=color, linestyle="--", alpha=0.7, label=label)

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "all_differences_cumulative.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create histogram of all eigenvalue differences across datasets"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display interactive plots, just save to files",
    )
    args = parser.parse_args()

    # Collect all differences
    all_diffs_df = collect_all_differences()

    if len(all_diffs_df) == 0:
        print("\nNo matched pairs data found.")
        return

    print(f"\nCollected {len(all_diffs_df)} total matched pairs across {all_diffs_df['dataset'].nunique()} datasets")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create plots
    print("\nGenerating plots...")
    create_histogram(all_diffs_df, OUTPUT_DIR, show_plots=not args.no_show)
    print(f"Plots saved to: {OUTPUT_DIR}")

    print("\nDone!")


if __name__ == "__main__":
    main()
