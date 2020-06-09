K-Truss
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program finds the k-truss for some k value in a given undirect graph.
A k-truss is the subgraph of a graph in which every edge in the subgraph
is a part of at least k - 2 triangles.

INPUT
--------------------------------------------------------------------------------

This application takes in symmetric Galois .gr graphs.
You must specify the -symmetricGraph flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/k-truss; make -j`

RUN
--------------------------------------------------------------------------------

The following are a few example command lines.

Find the 5 truss using 40 threads and the BSP algorithm.

-`$ ./k-truss-cpu <path-symmetric-clean-graph> -algo bsp -trussNum=5 -t 40 -symmetricGraph`

The following outputs the edges of a 10 truss to a file using bspJacobi (edge
removal is separated).

-`$ ./k-truss-cpu <path-symmetric-clean-graph> -algo bspJacobi -t 40 -trussNum=10 -o=10truss.out -symmetricGraph`

PERFORMANCE
--------------------------------------------------------------------------------

* The BSP variant (the default, -bsp) generally performs better in our experience.
