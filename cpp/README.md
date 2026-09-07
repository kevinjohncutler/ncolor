# The C++ engine

Everything in this directory except `binding.cpp` is header-only C++17
with no dependency beyond the standard library. `binding.cpp` is the
pybind11 module `ncolor._backend._impl`; the headers are the engine and
can be used on their own from C++ (or wrapped for C, Rust, Julia, R, or
a command-line tool) without Python anywhere in the picture.

## Using it without Python

```bash
cd cpp
c++ -std=c++17 -O3 -pthread -march=native -I. example_standalone.cpp -o example_standalone
./example_standalone
```

[`example_standalone.cpp`](example_standalone.cpp) runs the whole
pipeline on a synthetic image: connected components, Voronoi expansion,
the adjacency scan, and the coloring picker. [`ncolor.hpp`](ncolor.hpp)
includes every header; pick individual ones if you only need a kernel.

The pieces, in pipeline order:

| header | what it does |
|---|---|
| `threadpool.h`, `dispatch.hpp` | persistent fork-join pool with a wait-on-address idle path; atomic work-stealing chunk dispatch |
| `cc_label.hpp` | N-D connected-components labeling |
| `format_labels.hpp` | compact labels to `1..N`; range-checked casts from any integer or float type |
| `expand.hpp`, `expand_lp.hpp`, `chamfer.hpp` | N-D L1 / L2 Voronoi expansion (Saito-Toriwaki and Felzenszwalb sweeps) with NEON / SSE / AVX2 inner loops |
| `expand_clean.hpp` | the same expansion fused with bridge and stub removal (the `"clean"` mode) |
| `connect.hpp` | the `find_pairs` adjacency scan, hard and soft kernels in one pass |
| `color.hpp` | CSR construction, BFS / greedy coloring, repair, conflict check |
| `picker.hpp` | the coloring picker: the race of strategies (`tabucol.hpp`, `bb_dsatur.hpp`, `hea.hpp`, `clique_lb.hpp`) that `label` and `color_graph` run |
| `soft_color.hpp` | soft-constraint local search after the hard coloring |
| `delete_spurs.hpp`, `delete_spurs_labels.hpp`, `fast_despur.hpp` | spur and thin-bridge removal |
| `geometry.hpp`, `intrinsics.hpp` | N-D index helpers; portable bit intrinsics |

Design notes, benchmarks and the reasoning behind the kernels are in
[ARCHITECTURE.md](../ARCHITECTURE.md) at the repository root.

## Building the Python extension

`pip install -e .` from the repository root, or `python setup.py
build_ext --inplace` for an in-place build. Environment variables the
build honors:

| variable | effect |
|---|---|
| `NCOLOR_MARCH_NATIVE=0` | drop `-march=native` (on by default for source builds) |
| `NCOLOR_MARCH=x86-64-v2` | explicit `-march` when native is off; the x86_64 wheels use this |
| `NCOLOR_USE_CLANG_CL=1` | Windows: compile with clang-cl instead of cl.exe (needs LLVM on `PATH`) |
| `NCOLOR_NO_CALIBRATE=1` | skip the post-build SMT calibration (CI, cross builds) |

A build needs `pybind11` and a C++17 compiler; the resulting module is
loaded by `ncolor/_backend/__init__.py`, which also handles the
network-mounted-source case described in ARCHITECTURE.md.
