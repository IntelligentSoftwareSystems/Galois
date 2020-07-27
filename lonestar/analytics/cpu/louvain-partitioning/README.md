Partitioning using Louvain
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This directory contains a graph partitioning algorithm that uses a hierarchical community detection algorithm for coarsening. The algorithm uses the well-known multi-level partitioning approach, similar to GMetis.

INPUT
--------------------------------------------------------------------------------

This application takes in symmetric Galois .gr graphs.
You must specify the -symmetricGraph flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/louvain_partitioning; make -j`

RUN
--------------------------------------------------------------------------------

The following is an example command line.

-`$  ./louvain-partitioning-cpu -input <path-to-graph> -t 4 -tolerance=15 -threshold=0.000001 -max_iter 1000  -symmetricGraph`

