DESCRIPTION 
===========

This program finds the k-truss for some k value in a given undirect graph.
A k-truss is the subgraph of a graph in which every edge in the subgraph
is a part of at least k - 2 triangles.

INPUT
===========

All versions expect a symmetric graph in gr format. The graph must also
be clean (i.e., no self-loops, no edges).

BUILD
===========

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/ktruss; make -j`


RUN
===========

The following are a few example command lines.

Find the 5 truss using 40 threads and the BSP algorithm.

-`$ ./ktruss <path-symmetric-clean-graph> -algo bsp -trussNum=5 -t 40`

The following outputs the edges of a 10 truss to a file using bspJacobi (edge
removal is separated).

-`$ ./ktruss <path-symmetric-clean-graph> -algo bspJacobi -t 40 -trussNum=10 -o=10truss.out`

PERFORMANCE
===========

- In our experience, the BSP variant (the default, -bsp) performs best.
