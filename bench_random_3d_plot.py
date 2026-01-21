#!/usr/bin/env python3
"""
Benchmark ncolor.label on 3D spheres and 3D meshgrid labels.
Plots timing across a small range with a 2s guard.
"""

from __future__ import annotations

import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

os.environ.setdefault("NCOLOR_TRACE_FORMAT_LABELS", "0")

ROOT = pathlib.Path(__file__).resolve().parent

import ncolor  # noqa: E402

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = "#101010"
plt.rcParams["axes.facecolor"] = "#121212"
plt.rcParams["savefig.facecolor"] = "#101010"
plt.rcParams["axes.edgecolor"] = "#DDDDDD"
plt.rcParams["axes.labelcolor"] = "#FFFFFF"
plt.rcParams["xtick.color"] = "#FFFFFF"
plt.rcParams["ytick.color"] = "#FFFFFF"

MAX_BENCH_SECONDS = 2.0


def add_marker_legend(ax):
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="w",
               markersize=7, linestyle="None", label="<=4 colors"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="w", markeredgecolor="w",
               markersize=7, linestyle="None", label=">4 colors"),
    ]
    legend = ax.legend(handles=handles, loc="upper left", framealpha=0.2)
    ax.add_artist(legend)



def make_spheres(shape, n_spheres, rng):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    labels = np.zeros(shape, dtype=np.int32)
    radius_max = max(2, min(shape) // 6)
    max_attempts = n_spheres * 20
    attempts = 0
    label_id = 1
    while label_id <= n_spheres and attempts < max_attempts:
        attempts += 1
        radius = rng.integers(2, radius_max + 1)
        cz = rng.integers(radius, shape[0] - radius)
        cy = rng.integers(radius, shape[1] - radius)
        cx = rng.integers(radius, shape[2] - radius)
        mask = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
        if np.any(labels[mask] != 0):
            continue
        labels[mask] = label_id
        label_id += 1
    return labels


def make_meshgrid_labels(size):
    labels = np.arange(1, size ** 3 + 1, dtype=np.int32)
    return labels.reshape((size, size, size))


def time_coloring(fn, arr, iters=1, warmup=1, limit_ms=None):
    if limit_ms is None:
        limit_ms = MAX_BENCH_SECONDS * 1000.0
    for _ in range(warmup):
        t0 = time.perf_counter()
        fn(arr)
        if (time.perf_counter() - t0) * 1000.0 > limit_ms:
            return float("nan")
    t0 = time.perf_counter()
    fn(arr)
    dt = (time.perf_counter() - t0) * 1000.0
    if dt > limit_ms:
        return float("nan")
    return dt


def count_colors(arr):
    vals = np.unique(arr)
    if vals.size and vals[0] == 0:
        return int(vals.size - 1)
    return int(vals.size)


def main():
    rng = np.random.default_rng(3)
    sphere_sizes = [20, 24, 28, 32, 36, 40, 44, 48]
    mesh_sizes = [8, 10, 12, 14, 16, 18, 20]

    sphere_times = []
    sphere_colors = []
    sphere_valid = []

    mesh_times = []
    mesh_colors = []
    mesh_valid = []

    # Prewarm
    warm = make_spheres((20, 20, 20), 4, rng)
    _ = ncolor.label(warm, n=4, expand=False)

    for size in sphere_sizes:
        arr = make_spheres((size, size, size), max(4, size // 6), rng)
        fn = lambda a=arr: ncolor.label(a, n=6, expand=False)
        t = time_coloring(fn, arr, iters=1, warmup=1)
        if np.isnan(t):
            break
        out = ncolor.label(arr, n=6, expand=False)
        sphere_valid.append(size)
        sphere_times.append(t)
        sphere_colors.append(count_colors(out))
        print(f"spheres {size:3d}: {t:6.2f} ms ({sphere_colors[-1]})")

    for size in mesh_sizes:
        arr = make_meshgrid_labels(size)
        fn = lambda a=arr: ncolor.label(a, n=4, expand=False)
        t = time_coloring(fn, arr, iters=1, warmup=1)
        if np.isnan(t):
            break
        out = ncolor.label(arr, n=4, expand=False)
        mesh_valid.append(size)
        mesh_times.append(t)
        mesh_colors.append(count_colors(out))
        print(f"mesh {size:3d}: {t:6.2f} ms ({mesh_colors[-1]})")

    out_dir = ROOT / "bench_outputs"
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(sphere_valid, sphere_times, color="C0", label="3D spheres")
    sizes_le = [s for s, c in zip(sphere_valid, sphere_colors) if c <= 4]
    times_le = [t for t, c in zip(sphere_times, sphere_colors) if c <= 4]
    sizes_gt = [s for s, c in zip(sphere_valid, sphere_colors) if c > 4]
    times_gt = [t for t, c in zip(sphere_times, sphere_colors) if c > 4]
    if sizes_le:
        ax.scatter(sizes_le, times_le, marker="o", facecolors="none", edgecolors="C0", linewidths=1.6)
    if sizes_gt:
        ax.scatter(sizes_gt, times_gt, marker="^", facecolors="C0", edgecolors="C0", linewidths=1.0)
    ax.set_xlabel("Volume size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("3D Spheres")
    ax.grid(True, linestyle=":", alpha=0.6)
    add_marker_legend(ax)

    ax = axes[1]
    ax.plot(mesh_valid, mesh_times, color="C1", label="3D meshgrid")
    sizes_le = [s for s, c in zip(mesh_valid, mesh_colors) if c <= 4]
    times_le = [t for t, c in zip(mesh_times, mesh_colors) if c <= 4]
    sizes_gt = [s for s, c in zip(mesh_valid, mesh_colors) if c > 4]
    times_gt = [t for t, c in zip(mesh_times, mesh_colors) if c > 4]
    if sizes_le:
        ax.scatter(sizes_le, times_le, marker="o", facecolors="none", edgecolors="C1", linewidths=1.6)
    if sizes_gt:
        ax.scatter(sizes_gt, times_gt, marker="^", facecolors="C1", edgecolors="C1", linewidths=1.0)
    ax.set_xlabel("Volume size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("3D Meshgrid Labels")
    ax.grid(True, linestyle=":", alpha=0.6)
    add_marker_legend(ax)

    plt.tight_layout()
    out_path = out_dir / "random_3d_scaling.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
