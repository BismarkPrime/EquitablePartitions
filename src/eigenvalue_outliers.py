#!/usr/bin/env python3
"""
Eigenvalue Outliers Script

Collects eigenvalues with large differences (> threshold) between NumPy and LEParD
across ALL datasets and visualizes them together.

Usage:
    python eigenvalue_outliers.py [--threshold 1e-5] [--no-show]
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Configuration
DT_STATS_DIR = Path(__file__).parent.parent / "DT_Stats"
OUTPUT_DIR = Path(__file__).parent.parent / "eigenvalue_analysis" / "outliers"
DEFAULT_DIFF_THRESHOLD = 1e-5


def load_matched_pairs_bipartite(stats_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load matched_pairs_bipartite.csv and parse complex eigenvalues.

    Returns DataFrame with columns:
        numpy_eig, lepard_eig, difference, diff_magnitude,
        numpy_mag, lepard_mag (computed magnitudes)
    """
    csv_path = stats_dir / "matched_pairs_bipartite.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # Parse complex numbers from strings
    df["numpy_eig"] = df["numpy_eig"].apply(lambda x: complex(x))
    df["lepard_eig"] = df["lepard_eig"].apply(lambda x: complex(x))
    df["difference"] = df["difference"].apply(lambda x: complex(x))

    # Compute magnitudes
    df["numpy_mag"] = df["numpy_eig"].apply(abs)
    df["lepard_mag"] = df["lepard_eig"].apply(abs)

    # Ensure diff_magnitude is float
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


def collect_outliers(threshold: float) -> pd.DataFrame:
    """
    Collect all eigenvalues with diff_magnitude > threshold across all datasets.

    Returns DataFrame with additional 'dataset' column.
    """
    datasets = get_all_datasets()
    all_outliers = []

    print(
        f"Scanning {len(datasets)} datasets for outliers (threshold: {threshold:.0e})...\n"
    )

    for i, dataset_name in enumerate(datasets, 1):
        stats_dir = DT_STATS_DIR / f"{dataset_name}_stats"
        df = load_matched_pairs_bipartite(stats_dir)

        if df is None:
            continue

        # Filter for outliers
        outliers = df[df["diff_magnitude"] > threshold].copy()

        if len(outliers) > 0:
            outliers["dataset"] = dataset_name
            all_outliers.append(outliers)
            print(f"  [{i}/{len(datasets)}] {dataset_name}: {len(outliers)} outliers")
        else:
            print(f"  [{i}/{len(datasets)}] {dataset_name}: 0 outliers")

    if all_outliers:
        combined = pd.concat(all_outliers, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()


def compute_outlier_statistics(df: pd.DataFrame) -> Dict:
    """Compute summary statistics for outliers."""
    if len(df) == 0:
        return {"total_outliers": 0, "exact_matches": 0}

    diff_mags = df["diff_magnitude"]

    # Count exact matches (diff = 0)
    exact_matches = (diff_mags == 0).sum()

    # Count by dataset
    dataset_counts = df["dataset"].value_counts().to_dict()

    return {
        "total_outliers": len(df),
        "exact_matches": exact_matches,
        "num_datasets_with_outliers": df["dataset"].nunique(),
        "max_diff": diff_mags.max(),
        "mean_diff": diff_mags.mean(),
        "median_diff": diff_mags.median(),
        "min_diff": diff_mags.min(),
        "dataset_counts": dataset_counts,
    }


def format_outlier_stats_text(stats: Dict, threshold: float) -> str:
    """Format outlier statistics as text."""
    if stats["total_outliers"] == 0:
        return f"No outliers found (threshold: {threshold:.0e})"

    lines = [
        f"Eigenvalue Outlier Analysis",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Threshold: |difference| > {threshold:.0e}",
        "=" * 50,
        "",
        "Summary:",
        f"  Total outliers: {stats['total_outliers']}",
        f"  Exact matches (diff=0): {stats['exact_matches']}",
        f"  Datasets with outliers: {stats['num_datasets_with_outliers']}",
        "",
        "Difference Magnitudes:",
        f"  Max:    {stats['max_diff']:.6e}",
        f"  Mean:   {stats['mean_diff']:.6e}",
        f"  Median: {stats['median_diff']:.6e}",
        f"  Min:    {stats['min_diff']:.6e}",
        "",
        "Outliers by Dataset:",
    ]

    # Sort datasets by count (descending)
    sorted_counts = sorted(
        stats["dataset_counts"].items(), key=lambda x: x[1], reverse=True
    )
    for dataset, count in sorted_counts:
        lines.append(f"  {dataset}: {count}")

    return "\n".join(lines)


def create_outlier_plots(
    df: pd.DataFrame, output_dir: Path, threshold: float, show_plots: bool = True
) -> None:
    """Create all outlier analysis plots (aggregated and by-dataset versions)."""

    if len(df) == 0:
        print("No outliers to plot.")
        return

    # Prepare data - filter out zero differences for log scale plotting
    nonzero_mask = df["diff_magnitude"].values > 0
    df_nonzero = df[nonzero_mask]

    if len(df_nonzero) == 0:
        print("No non-zero outliers to plot.")
        return

    numpy_mags = df_nonzero["numpy_mag"].values
    lepard_mags = df_nonzero["lepard_mag"].values
    diff_mags = df_nonzero["diff_magnitude"].values
    datasets = df_nonzero["dataset"].values

    # Get unique datasets for coloring
    unique_datasets = df_nonzero["dataset"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_datasets)))
    dataset_color_map = {ds: colors[i] for i, ds in enumerate(unique_datasets)}

    # ===== Plot 1a: Diff vs NumPy (aggregated) =====
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(numpy_mags, diff_mags, alpha=0.6, s=20, color="blue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|NumPy eigenvalue|")
    ax.set_ylabel("|Difference|")
    ax.set_title(
        f"Outliers: Difference vs NumPy Magnitude (threshold > {threshold:.0e})"
    )
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"threshold={threshold:.0e}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "diff_vs_numpy.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # ===== Plot 1b: Diff vs NumPy (by dataset) =====
    fig, ax = plt.subplots(figsize=(12, 7))
    for dataset in unique_datasets:
        mask = datasets == dataset
        ax.scatter(
            numpy_mags[mask],
            diff_mags[mask],
            alpha=0.7,
            s=25,
            color=dataset_color_map[dataset],
            label=dataset,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|NumPy eigenvalue|")
    ax.set_ylabel("|Difference|")
    ax.set_title(f"Outliers by Dataset: Difference vs NumPy Magnitude")
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5)
    # Put legend outside if many datasets
    if len(unique_datasets) > 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "diff_vs_numpy_by_dataset.png", dpi=150, bbox_inches="tight"
    )
    if show_plots:
        plt.show()
    plt.close()

    # ===== Plot 2a: Diff vs LEParD (aggregated) =====
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(lepard_mags, diff_mags, alpha=0.6, s=20, color="blue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|LEParD eigenvalue|")
    ax.set_ylabel("|Difference|")
    ax.set_title(
        f"Outliers: Difference vs LEParD Magnitude (threshold > {threshold:.0e})"
    )
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"threshold={threshold:.0e}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "diff_vs_lepard.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # ===== Plot 2b: Diff vs LEParD (by dataset) =====
    fig, ax = plt.subplots(figsize=(12, 7))
    for dataset in unique_datasets:
        mask = datasets == dataset
        ax.scatter(
            lepard_mags[mask],
            diff_mags[mask],
            alpha=0.7,
            s=25,
            color=dataset_color_map[dataset],
            label=dataset,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|LEParD eigenvalue|")
    ax.set_ylabel("|Difference|")
    ax.set_title(f"Outliers by Dataset: Difference vs LEParD Magnitude")
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5)
    if len(unique_datasets) > 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "diff_vs_lepard_by_dataset.png", dpi=150, bbox_inches="tight"
    )
    if show_plots:
        plt.show()
    plt.close()

    # ===== Plot 4a: Histogram (aggregated) =====
    fig, ax = plt.subplots(figsize=(10, 6))
    log_range = np.log10(diff_mags.max()) - np.log10(diff_mags.min())
    if log_range > 2:
        bins = np.logspace(np.log10(diff_mags.min()), np.log10(diff_mags.max()), 50)
        ax.hist(diff_mags, bins=bins, alpha=0.7, color="purple", edgecolor="black")
        ax.set_xscale("log")
    else:
        ax.hist(diff_mags, bins=50, alpha=0.7, color="purple", edgecolor="black")
    ax.axvline(
        x=threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"threshold={threshold:.0e}",
    )
    ax.set_xlabel("|Difference|")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Outlier Difference Magnitudes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # ===== Plot 4b: Histogram (by dataset - stacked) =====
    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for stacked histogram
    log_range = np.log10(diff_mags.max()) - np.log10(diff_mags.min())
    if log_range > 2:
        bins = np.logspace(np.log10(diff_mags.min()), np.log10(diff_mags.max()), 30)
        use_log = True
    else:
        bins = 30
        use_log = False

    # Create list of arrays for each dataset
    hist_data = [
        df[df["dataset"] == ds]["diff_magnitude"].values for ds in unique_datasets
    ]
    hist_colors = [dataset_color_map[ds] for ds in unique_datasets]

    ax.hist(
        hist_data,
        bins=bins,
        alpha=0.7,
        stacked=True,
        label=unique_datasets,
        color=hist_colors,
        edgecolor="black",
        linewidth=0.5,
    )

    if use_log:
        ax.set_xscale("log")
    ax.axvline(x=threshold, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("|Difference|")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Outlier Differences by Dataset")
    if len(unique_datasets) > 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_by_dataset.png", dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Collect and visualize eigenvalue outliers across all datasets"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_DIFF_THRESHOLD,
        help=f"Difference threshold for outliers (default: {DEFAULT_DIFF_THRESHOLD})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display interactive plots, just save to files",
    )
    args = parser.parse_args()

    # Collect outliers
    outliers_df = collect_outliers(args.threshold)

    if len(outliers_df) == 0:
        print(f"\nNo outliers found with threshold {args.threshold:.0e}")
        return

    print(
        f"\nFound {len(outliers_df)} total outliers across {outliers_df['dataset'].nunique()} datasets"
    )

    # Compute statistics
    stats = compute_outlier_statistics(outliers_df)

    # Print statistics
    print("\n" + "=" * 60)
    stats_text = format_outlier_stats_text(stats, args.threshold)
    print(stats_text)
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save statistics
    stats_file = OUTPUT_DIR / "summary_stats.txt"
    with open(stats_file, "w") as f:
        f.write(stats_text)
    print(f"\nStatistics saved to: {stats_file}")

    # Save outliers CSV
    csv_file = OUTPUT_DIR / "outliers.csv"
    # Convert complex to string for CSV
    outliers_export = outliers_df.copy()
    outliers_export["numpy_eig"] = outliers_export["numpy_eig"].astype(str)
    outliers_export["lepard_eig"] = outliers_export["lepard_eig"].astype(str)
    outliers_export["difference"] = outliers_export["difference"].astype(str)
    outliers_export.to_csv(csv_file, index=False)
    print(f"Outliers CSV saved to: {csv_file}")

    # Create plots
    print("\nGenerating plots...")
    create_outlier_plots(
        outliers_df, OUTPUT_DIR, args.threshold, show_plots=not args.no_show
    )
    print(f"Plots saved to: {OUTPUT_DIR}")

    print("\nDone!")


if __name__ == "__main__":
    main()
