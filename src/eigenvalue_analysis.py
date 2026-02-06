#!/usr/bin/env python3
"""
Eigenvalue Analysis Script

Analyzes differences between eigenvalues computed by NumPy and those computed
by the LEParD algorithm. Generates plots and summary statistics for each dataset.

Usage:
    python eigenvalue_analysis.py [dataset1] [dataset2] ...

If no datasets are specified, iterates through all datasets in DT_Stats.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Configuration
DT_STATS_DIR = Path(__file__).parent.parent / "DT_Stats"
OUTPUT_DIR = Path(__file__).parent.parent / "eigenvalue_analysis"
ZERO_THRESHOLD = 1e-10  # Eigenvalues below this are considered "near-zero"


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

    # Ensure diff_magnitude is float (should already be, but just in case)
    df["diff_magnitude"] = df["diff_magnitude"].astype(float)

    return df


def compute_statistics(df: pd.DataFrame) -> Dict:
    """Compute summary statistics from the matched pairs DataFrame."""
    diff_mags = df["diff_magnitude"]
    numpy_mags = df["numpy_mag"]

    # Count near-zero eigenvalues
    near_zero_count = (numpy_mags < ZERO_THRESHOLD).sum()

    # Count exact matches
    exact_matches = (diff_mags == 0).sum()

    # Compute relative errors (excluding near-zero eigenvalues)
    mask = numpy_mags >= ZERO_THRESHOLD
    if mask.any():
        relative_errors = diff_mags[mask] / numpy_mags[mask]
        max_relative = relative_errors.max()
        mean_relative = relative_errors.mean()
        median_relative = relative_errors.median()
    else:
        max_relative = mean_relative = median_relative = float("nan")

    return {
        "total_count": len(df),
        "exact_matches": exact_matches,
        "near_zero_count": near_zero_count,
        "max_diff": diff_mags.max(),
        "mean_diff": diff_mags.mean(),
        "median_diff": diff_mags.median(),
        "max_relative_error": max_relative,
        "mean_relative_error": mean_relative,
        "median_relative_error": median_relative,
        "min_numpy_mag": numpy_mags.min(),
        "max_numpy_mag": numpy_mags.max(),
    }


def format_stats_text(dataset_name: str, stats: Dict) -> str:
    """Format statistics as a text string for display and saving."""
    lines = [
        f"Eigenvalue Analysis: {dataset_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        "",
        "Counts:",
        f"  Total eigenvalues:     {stats['total_count']}",
        f"  Exact matches (diff=0): {stats['exact_matches']}",
        f"  Near-zero eigenvalues: {stats['near_zero_count']} (|eig| < {ZERO_THRESHOLD})",
        "",
        "Absolute Differences:",
        f"  Max:    {stats['max_diff']:.6e}",
        f"  Mean:   {stats['mean_diff']:.6e}",
        f"  Median: {stats['median_diff']:.6e}",
        "",
        "Relative Errors (excluding near-zero eigenvalues):",
        f"  Max:    {stats['max_relative_error']:.6e}",
        f"  Mean:   {stats['mean_relative_error']:.6e}",
        f"  Median: {stats['median_relative_error']:.6e}",
        "",
        "Eigenvalue Magnitude Range:",
        f"  Min: {stats['min_numpy_mag']:.6e}",
        f"  Max: {stats['max_numpy_mag']:.6e}",
    ]
    return "\n".join(lines)


def print_statistics(dataset_name: str, stats: Dict) -> None:
    """Print summary statistics to console."""
    print(format_stats_text(dataset_name, stats))
    print()


def save_statistics(output_dir: Path, dataset_name: str, stats: Dict) -> None:
    """Save summary statistics to text file."""
    stats_text = format_stats_text(dataset_name, stats)
    stats_file = output_dir / "summary_stats.txt"
    with open(stats_file, "w") as f:
        f.write(stats_text)


def create_plots(
    df: pd.DataFrame, dataset_name: str, output_dir: Path, show_plots: bool = True
) -> None:
    """Create and save all analysis plots."""

    # Prepare data
    numpy_mags = df["numpy_mag"].values
    lepard_mags = df["lepard_mag"].values
    diff_mags = df["diff_magnitude"].values

    # Identify near-zero eigenvalues
    near_zero_numpy = numpy_mags < ZERO_THRESHOLD
    near_zero_lepard = lepard_mags < ZERO_THRESHOLD

    # For plotting, replace zeros with threshold value (so they show up on log scale)
    numpy_mags_plot = np.where(numpy_mags < ZERO_THRESHOLD, ZERO_THRESHOLD, numpy_mags)
    lepard_mags_plot = np.where(
        lepard_mags < ZERO_THRESHOLD, ZERO_THRESHOLD, lepard_mags
    )
    diff_mags_plot = np.where(diff_mags < ZERO_THRESHOLD, ZERO_THRESHOLD, diff_mags)

    # ===== Plot 1: Absolute Difference vs NumPy Magnitude =====
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot non-zero eigenvalues
    mask_normal = ~near_zero_numpy
    ax.scatter(
        numpy_mags_plot[mask_normal],
        diff_mags_plot[mask_normal],
        alpha=0.6,
        s=20,
        label="Normal eigenvalues",
        color="blue",
    )

    # Plot near-zero eigenvalues with different color
    if near_zero_numpy.any():
        ax.scatter(
            numpy_mags_plot[near_zero_numpy],
            diff_mags_plot[near_zero_numpy],
            alpha=0.6,
            s=20,
            label=f"Near-zero (|eig| < {ZERO_THRESHOLD})",
            color="red",
            marker="x",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|NumPy eigenvalue|")
    ax.set_ylabel("|Difference|")
    ax.set_title(f"{dataset_name}: Difference vs NumPy Eigenvalue Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "diff_vs_numpy.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # ===== Plot 2: Absolute Difference vs LEParD Magnitude =====
    fig, ax = plt.subplots(figsize=(10, 6))

    mask_normal = ~near_zero_lepard
    ax.scatter(
        lepard_mags_plot[mask_normal],
        diff_mags_plot[mask_normal],
        alpha=0.6,
        s=20,
        label="Normal eigenvalues",
        color="blue",
    )

    if near_zero_lepard.any():
        ax.scatter(
            lepard_mags_plot[near_zero_lepard],
            diff_mags_plot[near_zero_lepard],
            alpha=0.6,
            s=20,
            label=f"Near-zero (|eig| < {ZERO_THRESHOLD})",
            color="red",
            marker="x",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|LEParD eigenvalue|")
    ax.set_ylabel("|Difference|")
    ax.set_title(f"{dataset_name}: Difference vs LEParD Eigenvalue Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "diff_vs_lepard.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # ===== Plot 3: Relative Error vs NumPy Magnitude =====
    fig, ax = plt.subplots(figsize=(10, 6))

    # Only plot where numpy magnitude is not near-zero
    mask_valid = numpy_mags >= ZERO_THRESHOLD
    relative_errors = np.zeros_like(diff_mags)
    relative_errors[mask_valid] = diff_mags[mask_valid] / numpy_mags[mask_valid]

    # Replace zeros for log scale
    relative_errors_plot = np.where(
        relative_errors < ZERO_THRESHOLD, ZERO_THRESHOLD, relative_errors
    )

    ax.scatter(
        numpy_mags_plot[mask_valid],
        relative_errors_plot[mask_valid],
        alpha=0.6,
        s=20,
        color="green",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|NumPy eigenvalue|")
    ax.set_ylabel("Relative Error (|diff| / |eig|)")
    ax.set_title(f"{dataset_name}: Relative Error vs NumPy Eigenvalue Magnitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "relative_error.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # ===== Plot 4: Histogram of Difference Magnitudes =====
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out exact zeros for log histogram
    nonzero_diffs = diff_mags[diff_mags > 0]

    if len(nonzero_diffs) > 0:
        # Use log bins if data spans multiple orders of magnitude
        log_range = np.log10(nonzero_diffs.max()) - np.log10(nonzero_diffs.min())
        if log_range > 2:
            bins = np.logspace(
                np.log10(nonzero_diffs.min()), np.log10(nonzero_diffs.max()), 50
            )
            ax.hist(
                nonzero_diffs, bins=bins, alpha=0.7, color="purple", edgecolor="black"
            )
            ax.set_xscale("log")
        else:
            ax.hist(
                nonzero_diffs, bins=50, alpha=0.7, color="purple", edgecolor="black"
            )

    # Note exact matches
    exact_count = (diff_mags == 0).sum()
    if exact_count > 0:
        ax.axvline(x=ZERO_THRESHOLD, color="red", linestyle="--", alpha=0.7)
        ax.text(
            0.02,
            0.98,
            f"{exact_count} exact matches (diff=0)",
            transform=ax.transAxes,
            verticalalignment="top",
            color="red",
        )

    ax.set_xlabel("|Difference|")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset_name}: Distribution of Difference Magnitudes")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "histogram.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()


def get_all_datasets() -> List[str]:
    """Get list of all dataset names from DT_Stats directory."""
    if not DT_STATS_DIR.exists():
        return []

    datasets = []
    for item in DT_STATS_DIR.iterdir():
        if item.is_dir() and item.name.endswith("_stats"):
            # Check if matched_pairs_bipartite.csv exists
            if (item / "matched_pairs_bipartite.csv").exists():
                dataset_name = item.name[:-6]  # Remove "_stats" suffix
                datasets.append(dataset_name)

    return sorted(datasets)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze eigenvalue differences between NumPy and LEParD"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names to process. If none provided, processes all.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display interactive plots, just save to files",
    )
    args = parser.parse_args()

    # Determine which datasets to process
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = get_all_datasets()
        if not datasets:
            print(f"No datasets found in {DT_STATS_DIR}")
            print(
                "Make sure matched_pairs_bipartite.csv exists in the dataset stats folders."
            )
            return

    print(f"Found {len(datasets)} dataset(s) to process\n")

    # Process each dataset
    for i, dataset_name in enumerate(datasets, 1):
        print("=" * 60)
        print(f"[{i}/{len(datasets)}] Processing: {dataset_name}")
        print("=" * 60)

        stats_dir = DT_STATS_DIR / f"{dataset_name}_stats"

        if not stats_dir.exists():
            print(f"  Stats directory not found: {stats_dir}")
            print("  Skipping...\n")
            continue

        # Load data
        df = load_matched_pairs_bipartite(stats_dir)
        if df is None:
            print(f"  matched_pairs_bipartite.csv not found in {stats_dir}")
            print("  Skipping...\n")
            continue

        print(f"  Loaded {len(df)} matched eigenvalue pairs\n")

        # Compute and display statistics
        stats = compute_statistics(df)
        print_statistics(dataset_name, stats)

        # Create output directory
        dataset_output_dir = OUTPUT_DIR / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Save statistics
        save_statistics(dataset_output_dir, dataset_name, stats)
        print(f"  Statistics saved to: {dataset_output_dir / 'summary_stats.txt'}")

        # Create and save plots
        print(f"  Generating plots...")
        create_plots(df, dataset_name, dataset_output_dir, show_plots=not args.no_show)
        print(f"  Plots saved to: {dataset_output_dir}")

        # Prompt to continue (only if processing multiple datasets interactively)
        if len(datasets) > 1 and i < len(datasets) and not args.no_show:
            print()
            try:
                response = input("Continue to next dataset? [Y/n/q]: ").strip().lower()
                if response in ("n", "q", "quit", "exit"):
                    print("\nExiting early.")
                    break
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting.")
                break

        print()

    print("=" * 60)
    print("Analysis complete!")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
