/*
 * ncolor.hpp — umbrella header for the C++ engine, for use without Python.
 *
 * Everything under cpp/ except binding.cpp is header-only C++17 with no
 * dependency beyond the standard library and a thread pool (threadpool.h).
 * Include this one file and compile with -std=c++17 -O3 -pthread (add
 * -march=native or -march=x86-64-v2 for the SIMD paths):
 *
 *     #include "ncolor.hpp"
 *     ForkJoinPool pool(8);
 *     ncolor_cpp::ExpandBuffers bufs;
 *     ncolor_cpp::expand_labels_lp<2>(labels, labels, bufs, {H, W}, pool, 8);
 *
 * See example_standalone.cpp for the full pipeline (connected
 * components, adjacency, coloring) and README.md in this directory.
 */

#ifndef NCOLOR_HPP
#define NCOLOR_HPP

#include "threadpool.h"
#include "dispatch.hpp"
#include "geometry.hpp"
#include "intrinsics.hpp"

#include "cc_label.hpp"
#include "format_labels.hpp"
#include "expand.hpp"
#include "expand_lp.hpp"
#include "chamfer.hpp"
#include "expand_clean.hpp"
#include "connect.hpp"
#include "connect_with_face_count.hpp"
#include "color.hpp"
#include "picker.hpp"
#include "soft_color.hpp"
#include "delete_spurs.hpp"
#include "delete_spurs_labels.hpp"
#include "fast_despur.hpp"
#include "kempe_sa.hpp"

#endif  // NCOLOR_HPP
