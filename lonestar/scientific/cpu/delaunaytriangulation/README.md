DESCRIPTION 
===========

This program produces a Delaunay triangulation from a set of 2-D points. We 
implement the algorithm proposed by Bowyer and that by Watson:

1. Adrian Bowyer. Computing Dirichlet tessellations, The Computer Journal, 
Vol. 24, No. 2, pp 162 - 166, 1981.

2. David F. Watson. Computing the n-dimensional tessellation with application to 
Voronoi polytopes, The Computer Journal, Vol. 24, No. 2, pp 167 - 172, 1981. 


INPUT
===========

The implementations expect a list of nodes with their coordinates.


BUILD
===========

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/delaunaytriangulation; make -j`


RUN
===========

The following are a few example command lines.

-`$ ./delaunaytriangulation <path-to-node-list> -t 40`
-`$ ./delaunaytriangulation-det <path-to-node-list> -nondet -t 40`
-`$ ./delaunaytriangulation-det <path-to-node-list> -detBase -t 20`
-`$ ./delaunaytriangulation-det <path-to-node-list> -detPrefix -t 30`
-`$ ./delaunaytriangulation-det <path-to-node-list> -detDisjoint -t 15`


PERFORMANCE
===========

- In our experience, delaunaytriangulation outperforms deterministic variants in 
delaunaytriangulation-det.

- For the for_each loop named "Main", the chunk size of galois::wl<CA>() should be 
tuned. It controls the granularity of work distribution. The optimal value of the 
constant might depend on the architecture, so you might want to evaluate the 
performance over a range of values (say [16-4096]).
