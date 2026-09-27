"""Markdown tables for BENCHMARKS.md from release_comparison.py summaries.

Each host is given as NAME=DIR, where DIR holds ``t4/summary.json`` (four
workers), ``t1/summary.json`` (one worker), and ``concurrency.txt`` (the
output of ``concurrent_callers.py compare``). Ratios are other time divided
by current time, so values above 1 mean the current code is faster.

    python bench/release_tables.py "M5 Max=results/mac" "Ryzen 9 7950X=results/amd"
"""
import json
import re
import sys
from pathlib import Path

IMAGES = {
    "labels_logo": "logo, 241 x 205, 160 labels",
    "labels_synthetic800": "synthetic, 900 x 900, 682 labels",
    "labels_1024x1024": "sparse boxes, 1024 x 1024, 131 labels",
    "labels_2048x2048": "sparse boxes, 2048 x 2048, 523 labels",
    "labels_96x96x96": "boxes, 96 x 96 x 96, 50 labels",
    "labels_dense_128x128x128": "packed cells, 128 x 128 x 128, 2694 labels",
}
MASKS = {
    "mask_1024x1024": "1024 x 1024",
    "mask_2048x2048": "2048 x 2048",
    "mask_2x513x517": "2 x 513 x 517",
    "mask_96x96x96": "96 x 96 x 96",
}


def load(spec):
    name, directory = spec.split("=", 1)
    root = Path(directory)
    return name, {
        workers: json.loads((root / f"t{workers}" / "summary.json").read_text())
        for workers in (4, 1)
    }, (root / "concurrency.txt").read_text()


def ratio(summary, current_key, other_key, other_version):
    current = summary[current_key]
    other = summary[other_key]
    if other_version not in other["median_ms"]:
        return None
    if not all(v.get("valid", True) for v in other["validation"][other_version]):
        return None
    return other["median_ms"][other_version] / current["median_ms"]["current"]


def cell(value, digits=2, suffix="x"):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main(specs):
    hosts = [load(spec) for spec in specs]
    for workers in (4, 1):
        print(f"### label, {workers} worker{'s' if workers > 1 else ''}\n")
        rows = []
        for image, text in IMAGES.items():
            for name, runs, _ in hosts:
                s = runs[workers]
                rows.append([text, name, f"{s[image + '/default']['median_ms']['current']:.2f}",
                             f"{s[image + '/matched']['median_ms']['current']:.2f}",
                             cell(ratio(s, image + "/matched", image + "/matched", "1.5.3"))])
        print(table(["Image", "CPU", "Default ms", "Matched ms", "vs 1.5.3"], rows) + "\n")
    for workers in (4, 1):
        print(f"### expand_labels, {workers} worker{'s' if workers > 1 else ''}\n")
        rows = []
        for image, text in IMAGES.items():
            for name, runs, _ in hosts:
                s = runs[workers]
                key = image + "/expand"
                rows.append([text, name, f"{s[key]['median_ms']['current']:.2f}",
                             cell(ratio(s, key, key, "1.5.3")),
                             cell(ratio(s, key, image + "/scipy_feature", "external")),
                             cell(ratio(s, key, image + "/skimage_expand", "external"))])
        print(table(["Image", "CPU", "ncolor ms", "vs 1.5.3", "vs SciPy EDT", "vs scikit-image"], rows) + "\n")
    for workers in (4, 1):
        print(f"### regionprops, {workers} worker{'s' if workers > 1 else ''}\n")
        rows = []
        for image, text in IMAGES.items():
            for name, runs, _ in hosts:
                s = runs[workers]
                key = image + "/properties"
                rows.append([text, name, f"{s[key]['median_ms']['current']:.2f}",
                             cell(ratio(s, key, key, "external")),
                             cell(ratio(s, key, image + "/properties_table", "external"))])
        print(table(["Image", "CPU", "ncolor ms", "vs regionprops",
                     "vs regionprops_table"], rows) + "\n")
    print("### connected_components\n")
    rows = []
    names = sorted(k for k in hosts[0][1][4] if "/components_c" in k)
    for key in names:
        mask, conn = key.split("/components_c")
        stem, density = mask.rsplit("_", 1)
        for name, runs, _ in hosts:
            four, one = runs[4], runs[1]
            rows.append([MASKS[stem], f"{float(density):.0%}", conn, name,
                         f"{four[key]['median_ms']['current']:.2f}",
                         cell(ratio(four, key, key, "external")),
                         f"{one[key]['median_ms']['current']:.2f}",
                         cell(ratio(one, key, key, "external"))])
    print(table(["Mask", "Fill", "Conn", "CPU", "4 workers ms", "vs scikit-image",
                 "1 worker ms", "vs scikit-image"], rows) + "\n")
    print("### Four concurrent callers\n")
    rows = []
    pattern = re.compile(r"(\d+)\s+engines\s+([\d.]+) ms\s+turns\s+([\d.]+) ms\s+speedup ([\d.]+)x")
    for name, _, text in hosts:
        for size, engines, turns, speedup in pattern.findall(text):
            rows.append([f"{size} x {size}", name, engines, turns, f"{speedup}x"])
    rows.sort(key=lambda r: int(r[0].split()[0]))
    print(table(["Image", "CPU", "Overlapping ms", "Taking turns ms", "Speedup"], rows))


if __name__ == "__main__":
    main(sys.argv[1:])
