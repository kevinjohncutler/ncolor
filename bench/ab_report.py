"""Pair the A/B runs and put an interval on the difference.

Reads the JSON files ab_bench.sh writes and reports, per host, the
speedup of the candidate over the baseline with a bootstrap confidence
interval.

The experimental unit is the repetition, not the call. Calls inside one
process are not independent: a process can settle into a slow state and
stay there for its whole run, which was measured on an i9 where the same
case came out at 0.33 ms in some processes and 0.7 ms in others, in both
builds alike. Pooling every call would treat that as ordinary spread,
give an interval several times too narrow, and report whichever build
happened to draw more slow processes as a regression. So each repetition
is reduced to one number, the median of its calls, and the comparison
and the bootstrap both run over repetitions.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

OPS = ("label", "label_noexpand", "expand_labels", "connect", "format_labels")
BOOT = 4000


def load(host_dir):
    """{mode: {build: {case: {op: [one number per repetition]}}}}.

    Each repetition contributes the median of its own calls, so a
    process that ran slow throughout counts once rather than forty
    times.
    """
    data = {m: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for m in ("serial", "concurrent", "graphs")}
    for p in sorted(Path(host_dir).glob("*.json")):
        if p.name.startswith("verify"):
            continue
        d = json.loads(p.read_text())
        mode, build = d["meta"]["mode"], d["meta"]["build"]
        for case, ops in d["results"].items():
            for op, samples in ops.items():
                if samples:
                    data[mode][build][case][op].append(statistics.median(samples))
    return data


def boot_ratio(base, cand, rng, higher_is_better=False):
    """Speedup and a 95% interval, by resampling repetitions."""
    b, c = np.asarray(base), np.asarray(cand)
    point = (statistics.median(base) / statistics.median(cand) if not higher_is_better
             else statistics.median(cand) / statistics.median(base))
    idx_b = rng.integers(0, len(b), size=(BOOT, len(b)))
    idx_c = rng.integers(0, len(c), size=(BOOT, len(c)))
    mb, mc = np.median(b[idx_b], axis=1), np.median(c[idx_c], axis=1)
    draws = mb / mc if not higher_is_better else mc / mb
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def verdict(lo, hi, band=0.02):
    if lo > 1 + band:
        return "faster"
    if hi < 1 - band:
        return "SLOWER"
    return "same"


def report(root, detail=False, pairs=(("base", "cand"),)):
    rng = np.random.default_rng(0)
    hosts = sorted(p for p in Path(root).iterdir() if p.is_dir())
    all_rows = []

    for h in hosts:
        data = load(h)
        for a_name, b_name in pairs:
            report_pair(h, data, a_name, b_name, rng, detail, all_rows,
                        len(pairs) > 1)
    summarize(all_rows)


def summarize(all_rows):
    if not all_rows:
        return
    by_pair = defaultdict(list)
    for row in all_rows:
        by_pair[row[0]].append(row)
    print("\n### All hosts together\n")
    print("| comparison | measurements | geometric mean | faster | slower |")
    print("|---|---|---|---|---|")
    for pair, rows in by_pair.items():
        r = np.array([x[6] for x in rows])
        print(f"| {pair} | {len(rows)} | {np.exp(np.mean(np.log(r))):.3f}x | "
              f"{sum(1 for x in rows if verdict(x[7], x[8]) == 'faster')} | "
              f"{sum(1 for x in rows if verdict(x[7], x[8]) == 'SLOWER')} |")


def report_pair(h, data, a_name, b_name, rng, detail, all_rows, multi):
    """Speedup of b over a on one host."""
    if True:
        base, cand = data["serial"].get(a_name), data["serial"].get(b_name)
        label = f"{a_name} -> {b_name}"
        if not base or not cand:
            if not multi:
                print(f"\n### {h.name}: incomplete\n")
            return
        reps = {c: len(v[OPS[0]]) for c, v in base.items()}
        print(f"\n### {h.name}: {label}  ({min(reps.values())}-{max(reps.values())} "
              f"repetitions per case and op, per build)\n")

        rows = []
        for case in sorted(base):
            for op in OPS:
                if op not in base[case] or op not in cand[case]:
                    continue
                pt, lo, hi = boot_ratio(base[case][op], cand[case][op], rng)
                rows.append((case, op, statistics.median(base[case][op]),
                             statistics.median(cand[case][op]), pt, lo, hi))
        all_rows += [(label, h.name) + r for r in rows]

        ratios = np.array([r[4] for r in rows])
        gm = float(np.exp(np.mean(np.log(ratios))))
        idx = rng.integers(0, len(ratios), size=(BOOT, len(ratios)))
        gdraw = np.exp(np.mean(np.log(ratios[idx]), axis=1))
        glo, ghi = np.percentile(gdraw, [2.5, 97.5])
        slower = [r for r in rows if verdict(r[5], r[6]) == "SLOWER"]
        faster = [r for r in rows if verdict(r[5], r[6]) == "faster"]

        print(f"| metric | value |")
        print(f"|---|---|")
        print(f"| measurements (case x op) | {len(rows)} |")
        print(f"| geometric mean speedup | {gm:.3f}x (95% CI {glo:.3f} to {ghi:.3f}) |")
        print(f"| faster beyond noise | {len(faster)} |")
        print(f"| slower beyond noise | {len(slower)} |")
        print(f"| within noise | {len(rows) - len(faster) - len(slower)} |")
        print(f"| best | {max(rows, key=lambda r: r[4])[0]}/{max(rows, key=lambda r: r[4])[1]} "
              f"{max(ratios):.2f}x |")
        print(f"| worst | {min(rows, key=lambda r: r[4])[0]}/{min(rows, key=lambda r: r[4])[1]} "
              f"{min(ratios):.2f}x |")

        if slower:
            print(f"\nRegressions on {h.name} ({label}):\n")
            print("| case | op | base ms | cand ms | speedup | 95% CI |")
            print("|---|---|---|---|---|---|")
            for c, o, b, cd, pt, lo, hi in sorted(slower, key=lambda r: r[4]):
                print(f"| {c} | {o} | {b:.3f} | {cd:.3f} | {pt:.3f}x | {lo:.3f} to {hi:.3f} |")

        cb, cc = data["concurrent"].get(a_name), data["concurrent"].get(b_name)
        if cb and cc:
            crows = []
            for case in sorted(cb):
                pt, lo, hi = boot_ratio(cb[case]["label_rate"], cc[case]["label_rate"],
                                        rng, higher_is_better=True)
                crows.append((case, pt, lo, hi))
            cr = np.array([r[1] for r in crows])
            print(f"\n4 concurrent callers, images per second: geometric mean "
                  f"{np.exp(np.mean(np.log(cr))):.2f}x, "
                  f"range {cr.min():.2f}x to {cr.max():.2f}x")

        if detail:
            print(f"\n<details><summary>every measurement on {h.name}</summary>\n")
            print("| case | op | base ms | cand ms | speedup | 95% CI | verdict |")
            print("|---|---|---|---|---|---|---|")
            for c, o, b, cd, pt, lo, hi in rows:
                print(f"| {c} | {o} | {b:.3f} | {cd:.3f} | {pt:.3f}x | "
                      f"{lo:.3f} to {hi:.3f} | {verdict(lo, hi)} |")
            print("\n</details>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench_outputs/ab")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--pairs", default="base:cand",
                    help="comma-separated a:b comparisons, e.g. base:mid,mid:cand")
    a = ap.parse_args()
    report(a.dir, a.detail,
           tuple(tuple(x.split(":")) for x in a.pairs.split(",")))
