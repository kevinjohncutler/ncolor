"""Digest of ncolor.label's output for every corpus case.

Run on several machines and diff the JSON: same graph in, same coloring
out, or not. Colorings are compared in full, not just the foreground,
because the question is bit reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket

import numpy as np


def digest(a):
    return hashlib.blake2b(np.ascontiguousarray(np.asarray(a)).tobytes(),
                           digest_size=12).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 uses the machine's own thread count")
    args = ap.parse_args()

    import ncolor
    from ncolor._backend import _smt
    npz = np.load(args.corpus)
    eng = ncolor.Engine(n_threads=args.threads) if args.threads else None
    lab = (lambda m: eng.label(m)) if eng else ncolor.label

    res = {}
    for name in npz.files:
        m = npz[name]
        out, n = (eng.label(m, return_n=True) if eng
                  else ncolor.label(m, return_n=True))
        res[name] = {"coloring": digest(out), "n": int(n)}

    meta = {
        "host": socket.gethostname().split(".")[0],
        "threads": args.threads or _smt.auto_threads(),
        "deterministic": bool(os.environ.get("NCOLOR_DETERMINISTIC")),
        "ncolor": getattr(ncolor, "__version__", "?"),
    }
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "results": res}, fh, indent=1)
    print(f"[{meta['host']}] threads={meta['threads']} "
          f"deterministic={meta['deterministic']}", flush=True)


if __name__ == "__main__":
    main()
