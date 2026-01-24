#!/usr/bin/env python3
"""
Eigenvalue Rematch Script

Recomputes eigenvalue matching using scipy's linear_sum_assignment (Hungarian algorithm)
for optimal min-cost bipartite matching. Compares results with the original greedy matching.

Usage:
    python eigenvalue_rematch.py [dataset1] [dataset2] ...
    python eigenvalue_rematch.py --threshold 1e-4

If no datasets specified, processes all datasets with required files.
"""

import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# Configuration
DT_STATS_DIR = Path(__file__).parent.parent / "DT_Stats"
DEFAULT_MIN_THRESHOLD = 1e-8
DEFAULT_MAX_THRESHOLD = 1e-2


def load_lepard_eigenvalues(stats_dir: Path) -> Tuple[List[complex], Optional[str]]:
    """Load eigenvalues from globals.txt and locals.txt."""
    eigenvalues = []

    globals_file = stats_dir / "globals.txt"
    locals_file = stats_dir / "locals.txt"

    if not globals_file.exists():
        return [], f"globals.txt not found"
    if not locals_file.exists():
        return [], f"locals.txt not found"

    try:
        with open(globals_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    eigenvalues.append(complex(line))
    except Exception as e:
        return [], f"Error parsing globals.txt: {e}"

    try:
        with open(locals_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    eigenvalues.append(complex(line))
    except Exception as e:
        return [], f"Error parsing locals.txt: {e}"

    return eigenvalues, None


def load_numpy_eigenvalues(stats_dir: Path) -> Tuple[List[complex], Optional[str]]:
    """Load eigenvalues from numpy_eigenvalues.txt."""
    numpy_file = stats_dir / "numpy_eigenvalues.txt"

    if not numpy_file.exists():
        return [], "numpy_eigenvalues.txt not found"

    eigenvalues = []
    try:
        with open(numpy_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    eigenvalues.append(complex(line))
    except Exception as e:
        return [], f"Error parsing numpy_eigenvalues.txt: {e}"

    return eigenvalues, None


def load_old_matching(stats_dir: Path) -> Optional[pd.DataFrame]:
    """Load the original matched_pairs.csv for comparison."""
    csv_path = stats_dir / "matched_pairs.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        df["diff_magnitude"] = df["diff_magnitude"].astype(float)
        return df
    except Exception:
        return None


def match_at_threshold(
    numpy_eigs: List[complex], lepard_eigs: List[complex], threshold: float
) -> Tuple[List[Tuple[int, int, complex, complex, float]], List[int], List[int]]:
    """
    Run single round of optimal matching at given threshold.

    Returns:
        matches: List of (numpy_idx, lepard_idx, numpy_eig, lepard_eig, diff_magnitude)
        unmatched_numpy_indices: indices not matched
        unmatched_lepard_indices: indices not matched
    """
    n_numpy = len(numpy_eigs)
    n_lepard = len(lepard_eigs)

    if n_numpy == 0 or n_lepard == 0:
        return [], list(range(n_numpy)), list(range(n_lepard))

    # Build cost matrix
    n_max = max(n_numpy, n_lepard)
    cost_matrix = np.full((n_max, n_max), math.inf)

    for i in range(n_numpy):
        for j in range(n_lepard):
            dist = abs(numpy_eigs[i] - lepard_eigs[j])
            if dist <= threshold:
                cost_matrix[i, j] = dist

    # Check if there are any valid matches
    if np.all(np.isinf(cost_matrix)):
        # No valid matches at this threshold
        return [], list(range(n_numpy)), list(range(n_lepard))

    # Solve assignment problem
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        # Assignment infeasible (can happen with partial inf matrices)
        return [], list(range(n_numpy)), list(range(n_lepard))

    # Extract matches
    matches = []
    matched_numpy = set()
    matched_lepard = set()

    for i, j in zip(row_ind, col_ind):
        if i < n_numpy and j < n_lepard and cost_matrix[i, j] < math.inf:
            diff_mag = abs(numpy_eigs[i] - lepard_eigs[j])
            matches.append((i, j, numpy_eigs[i], lepard_eigs[j], diff_mag))
            matched_numpy.add(i)
            matched_lepard.add(j)

    unmatched_numpy = [i for i in range(n_numpy) if i not in matched_numpy]
    unmatched_lepard = [j for j in range(n_lepard) if j not in matched_lepard]

    return matches, unmatched_numpy, unmatched_lepard


def compute_optimal_matching(
    numpy_eigs: List[complex],
    lepard_eigs: List[complex],
    min_threshold: float = DEFAULT_MIN_THRESHOLD,
    max_threshold: float = DEFAULT_MAX_THRESHOLD,
) -> Tuple[List[Tuple[complex, complex, complex, float]], List[Tuple[str, complex]]]:
    """
    Compute optimal bipartite matching using iterative threshold approach.

    Starts with min_threshold and increases by 10x each iteration until
    max_threshold is reached or all eigenvalues are matched.

    Returns:
        matched_pairs: List of (numpy_eig, lepard_eig, difference, diff_magnitude)
        unmatched: List of (source, eigenvalue) for unmatched eigenvalues
    """
    if len(numpy_eigs) == 0 or len(lepard_eigs) == 0:
        unmatched = []
        unmatched.extend([("numpy", e) for e in numpy_eigs])
        unmatched.extend([("lepard", e) for e in lepard_eigs])
        return [], unmatched

    # Track remaining unmatched eigenvalues
    remaining_numpy = list(enumerate(numpy_eigs))  # (original_idx, value)
    remaining_lepard = list(enumerate(lepard_eigs))

    all_matched_pairs = []
    threshold = min_threshold

    while threshold <= max_threshold and remaining_numpy and remaining_lepard:
        # Extract just the values for matching
        numpy_vals = [v for _, v in remaining_numpy]
        lepard_vals = [v for _, v in remaining_lepard]

        # Run matching at current threshold
        matches, unmatched_n_idx, unmatched_l_idx = match_at_threshold(
            numpy_vals, lepard_vals, threshold
        )

        # Record matched pairs
        for local_n, local_l, numpy_eig, lepard_eig, diff_mag in matches:
            difference = numpy_eig - lepard_eig
            all_matched_pairs.append((numpy_eig, lepard_eig, difference, diff_mag))

        # Update remaining to only unmatched
        remaining_numpy = [remaining_numpy[i] for i in unmatched_n_idx]
        remaining_lepard = [remaining_lepard[j] for j in unmatched_l_idx]

        # Increase threshold for next iteration
        threshold *= 10

    # Collect final unmatched
    unmatched = []
    for _, eig in remaining_numpy:
        unmatched.append(("numpy", eig))
    for _, eig in remaining_lepard:
        unmatched.append(("lepard", eig))

    return all_matched_pairs, unmatched


def save_matching(
    stats_dir: Path,
    matched_pairs: List[Tuple[complex, complex, complex, float]],
    unmatched: List[Tuple[str, complex]],
) -> None:
    """Save the new matching to matched_pairs_bipartite.csv."""
    # Save matched pairs
    matched_file = stats_dir / "matched_pairs_bipartite.csv"
    with open(matched_file, "w") as f:
        f.write("numpy_eig,lepard_eig,difference,diff_magnitude\n")
        for numpy_eig, lepard_eig, diff, mag in matched_pairs:
            f.write(f"{numpy_eig},{lepard_eig},{diff},{mag}\n")

    # Save unmatched
    unmatched_file = stats_dir / "unmatched_bipartite.csv"
    with open(unmatched_file, "w") as f:
        f.write("source,eigenvalue\n")
        for source, eig in unmatched:
            f.write(f"{source},{eig}\n")


def compare_matchings(
    old_df: Optional[pd.DataFrame],
    new_matched: List[Tuple[complex, complex, complex, float]],
) -> dict:
    """Compare old greedy matching with new optimal matching."""
    result = {}

    # New matching stats
    if new_matched:
        new_diffs = [m[3] for m in new_matched]
        result["new_count"] = len(new_matched)
        result["new_max"] = max(new_diffs)
        result["new_mean"] = sum(new_diffs) / len(new_diffs)
        result["new_median"] = sorted(new_diffs)[len(new_diffs) // 2]
    else:
        result["new_count"] = 0
        result["new_max"] = result["new_mean"] = result["new_median"] = float("nan")

    # Old matching stats
    if old_df is not None and len(old_df) > 0:
        old_diffs = old_df["diff_magnitude"].values
        result["old_count"] = len(old_df)
        result["old_max"] = old_diffs.max()
        result["old_mean"] = old_diffs.mean()
        result["old_median"] = np.median(old_diffs)
        result["has_old"] = True
    else:
        result["old_count"] = 0
        result["old_max"] = result["old_mean"] = result["old_median"] = float("nan")
        result["has_old"] = False

    return result


def print_comparison(dataset_name: str, stats: dict, n_unmatched: int) -> None:
    """Print comparison between old and new matching."""
    print(f"\n  Matched pairs: {stats['new_count']}, Unmatched: {n_unmatched}")

    if stats["has_old"]:
        print(f"\n  Old matching (greedy):")
        print(f"    Count:  {stats['old_count']}")
        print(f"    Max:    {stats['old_max']:.6e}")
        print(f"    Mean:   {stats['old_mean']:.6e}")
        print(f"    Median: {stats['old_median']:.6e}")

    print(f"\n  New matching (optimal bipartite):")
    print(f"    Count:  {stats['new_count']}")
    print(f"    Max:    {stats['new_max']:.6e}")
    print(f"    Mean:   {stats['new_mean']:.6e}")
    print(f"    Median: {stats['new_median']:.6e}")

    if stats["has_old"] and stats["new_count"] > 0:
        # Compare improvements
        if stats["new_max"] < stats["old_max"]:
            improvement = (stats["old_max"] - stats["new_max"]) / stats["old_max"] * 100
            print(f"\n  Max diff improved by {improvement:.1f}%")
        elif stats["new_max"] > stats["old_max"]:
            print(f"\n  Max diff got WORSE (old was better)")
        else:
            print(f"\n  Max diff unchanged")


def get_all_datasets() -> List[str]:
    """Get list of all dataset names that have required files."""
    if not DT_STATS_DIR.exists():
        return []

    datasets = []
    for item in DT_STATS_DIR.iterdir():
        if item.is_dir() and item.name.endswith("_stats"):
            # Check if required files exist
            has_globals = (item / "globals.txt").exists()
            has_locals = (item / "locals.txt").exists()
            has_numpy = (item / "numpy_eigenvalues.txt").exists()

            if has_globals and has_locals and has_numpy:
                dataset_name = item.name[:-6]  # Remove "_stats" suffix
                datasets.append(dataset_name)

    return sorted(datasets)


def main():
    parser = argparse.ArgumentParser(
        description="Recompute eigenvalue matching using optimal bipartite assignment"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names to process. If none provided, processes all.",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=DEFAULT_MIN_THRESHOLD,
        help=f"Starting threshold (default: {DEFAULT_MIN_THRESHOLD})",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=DEFAULT_MAX_THRESHOLD,
        help=f"Maximum threshold (default: {DEFAULT_MAX_THRESHOLD})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if matched_pairs_bipartite.csv already exists",
    )
    args = parser.parse_args()

    # Determine which datasets to process
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = get_all_datasets()
        if not datasets:
            print(f"No datasets found in {DT_STATS_DIR}")
            print("Make sure globals.txt, locals.txt, and numpy_eigenvalues.txt exist.")
            return

    print(f"Processing {len(datasets)} dataset(s)")
    print(f"Threshold range: {args.min_threshold:.0e} -> {args.max_threshold:.0e} (10x increments)\n")

    # Summary counters
    total_improved = 0
    total_same = 0
    total_worse = 0
    total_no_comparison = 0
    total_skipped = 0

    for i, dataset_name in enumerate(datasets, 1):
        print("=" * 60)
        print(f"[{i}/{len(datasets)}] {dataset_name}")

        stats_dir = DT_STATS_DIR / f"{dataset_name}_stats"

        if not stats_dir.exists():
            print(f"  Stats directory not found: {stats_dir}")
            continue

        # Skip if already computed (unless --force)
        bipartite_file = stats_dir / "matched_pairs_bipartite.csv"
        if bipartite_file.exists() and not args.force:
            print(f"  Skipping (already exists, use --force to recompute)")
            total_skipped += 1
            continue

        # Load eigenvalues
        lepard_eigs, err = load_lepard_eigenvalues(stats_dir)
        if err:
            print(f"  Error loading LEParD eigenvalues: {err}")
            continue

        numpy_eigs, err = load_numpy_eigenvalues(stats_dir)
        if err:
            print(f"  Error loading NumPy eigenvalues: {err}")
            continue

        print(f"  Loaded {len(numpy_eigs)} NumPy, {len(lepard_eigs)} LEParD eigenvalues")

        # Load old matching for comparison
        old_df = load_old_matching(stats_dir)

        # Compute optimal matching
        matched_pairs, unmatched = compute_optimal_matching(
            numpy_eigs, lepard_eigs, args.min_threshold, args.max_threshold
        )

        # Compare matchings
        comparison = compare_matchings(old_df, matched_pairs)
        print_comparison(dataset_name, comparison, len(unmatched))

        # Track improvement
        if comparison["has_old"] and comparison["new_count"] > 0:
            if comparison["new_max"] < comparison["old_max"]:
                total_improved += 1
            elif comparison["new_max"] > comparison["old_max"]:
                total_worse += 1
            else:
                total_same += 1
        else:
            total_no_comparison += 1

        # Save new matching
        save_matching(stats_dir, matched_pairs, unmatched)
        print(f"\n  Saved: matched_pairs_bipartite.csv, unmatched_bipartite.csv")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total datasets: {len(datasets)}")
    if total_skipped > 0:
        print(f"  Skipped (already computed): {total_skipped}")
    if total_improved + total_same + total_worse > 0:
        print(f"  Improved (lower max diff): {total_improved}")
        print(f"  Same: {total_same}")
        print(f"  Worse: {total_worse}")
    if total_no_comparison > 0:
        print(f"  No comparison available: {total_no_comparison}")


if __name__ == "__main__":
    main()
