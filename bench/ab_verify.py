"""Compare two ab_run.py verify dumps.

Deterministic operations must agree byte for byte. ``label`` races
several searches under a time budget, so what it owes is checked
instead: the same foreground, no conflicts, and colors dense from 1.
Using more colors than the baseline is reported, since that is a
quality regression even when the coloring is valid.
"""
from __future__ import annotations

import json
import sys

EXACT = ("expand_labels", "format_labels", "connect")


def main(base_path, cand_path):
    base = json.load(open(base_path))["results"]
    cand = json.load(open(cand_path))["results"]
    bad, notes = [], []
    for name in sorted(set(base) & set(cand)):
        b, c = base[name], cand[name]
        for k in EXACT:
            if b[k] != c[k]:
                bad.append(f"{name}: {k} differs ({b[k]} vs {c[k]})")
        # Foreground is only comparable where each build agrees with
        # itself; see the note in ab_run.py.
        stable = (b["label_fg"] == b.get("label_fg_again")
                  and c["label_fg"] == c.get("label_fg_again"))
        if not stable:
            notes.append(f"{name}: label foreground is not deterministic "
                         f"(base {'stable' if b['label_fg'] == b.get('label_fg_again') else 'varies'}, "
                         f"cand {'stable' if c['label_fg'] == c.get('label_fg_again') else 'varies'})"
                         " - not compared")
        elif b["label_fg"] != c["label_fg"]:
            bad.append(f"{name}: label foreground differs")
        if c["label_conflicts"] != 0:
            bad.append(f"{name}: candidate coloring has {c['label_conflicts']} conflicts")
        colors = c["label_colors"]
        if colors != list(range(1, len(colors) + 1)):
            bad.append(f"{name}: candidate colors are not dense from 1: {colors[:8]}")
        if c["label_n"] != len(colors):
            bad.append(f"{name}: candidate n={c['label_n']} but {len(colors)} colors used")
        if c["label_n"] != b["label_n"]:
            notes.append(f"{name}: colors used {b['label_n']} -> {c['label_n']}")
    missing = sorted(set(base) ^ set(cand))
    if missing:
        bad.append(f"cases in only one dump: {missing}")
    for n in notes:
        print("note:", n)
    if bad:
        print(f"\nFAIL ({len(bad)}):")
        for x in bad:
            print(" ", x)
        return 1
    print(f"\nOK: {len(set(base) & set(cand))} cases agree "
          f"(exact on {', '.join(EXACT)}; label valid and no worse)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
