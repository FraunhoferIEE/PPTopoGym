"""Diff two ``dump_results.py`` archives and report where -- and by how much -- they disagree.

Physical quantities are compared as absolute differences with NaN treated as a *value*: a NaN
that appears on one side only is a disagreement about whether the grid is solvable there, which
matters far more than a small numeric delta, so it is reported separately from the magnitudes.

Run with::

    python diff_results.py a.npz b.npz --labels develop,lightsim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compare(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int]:
    """Compare two arrays elementwise, ignoring positions that are NaN on both sides.

    :param left: the reference values.
    :param right: the values under test.
    :return: (largest absolute difference, count of NaN-mask disagreements, compared elements).
    """
    if left.shape != right.shape:
        return float("inf"), -1, 0
    left_nan, right_nan = np.isnan(left), np.isnan(right)
    mask_mismatch = int(np.count_nonzero(left_nan ^ right_nan))
    both = ~(left_nan | right_nan)
    delta = float(np.max(np.abs(left[both] - right[both]))) if np.any(both) else 0.0
    return delta, mask_mismatch, int(np.count_nonzero(both))


def main() -> None:
    """Print a per-quantity worst-case table plus every decision-level disagreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--labels", default="left,right")
    args = parser.parse_args()
    left_label, right_label = args.labels.split(",")

    left, right = np.load(args.left), np.load(args.right)
    shared = [key for key in left.files if key in right.files]
    missing = sorted(set(left.files) ^ set(right.files))

    worst: dict[str, tuple[float, int, str]] = {}
    decisions: list[str] = []
    for key in shared:
        quantity = key.rsplit("/", 1)[-1]
        delta, mask_mismatch, _n = compare(left[key], right[key])
        previous = worst.get(quantity, (-1.0, 0, ""))
        if delta > previous[0]:
            worst[quantity] = (delta, previous[1] + mask_mismatch, key)
        else:
            worst[quantity] = (previous[0], previous[1] + mask_mismatch, previous[2])
        if quantity == "converged" and delta > 0:
            decisions.append(f"  {key}: {left_label}={left[key][0]} {right_label}={right[key][0]}")

    print(f"{left_label} vs {right_label}: {len(shared)} arrays compared")  # noqa: T201
    if missing:
        print(f"  !! only on one side: {missing}")  # noqa: T201
    print(f"{'quantity':<16} {'worst abs diff':>16} {'NaN-mask diffs':>15}  worst at")  # noqa: T201
    print("-" * 76)  # noqa: T201
    for quantity, (delta, mask_mismatch, where) in sorted(worst.items()):
        print(f"{quantity:<16} {delta:16.3e} {mask_mismatch:15d}  {where}")  # noqa: T201
    if decisions:
        print("\nconvergence decisions that differ:")  # noqa: T201
        print("\n".join(decisions))  # noqa: T201


if __name__ == "__main__":
    main()
