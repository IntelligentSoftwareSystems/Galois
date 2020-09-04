Triangle Counting
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------


This benchmark counts the number of triangles in a given undirected graph. It implements the approach from Polak [1] in IrGL[2].

[1] Adam Polak. Counting triangles in large graphs on GPU. In IPDPS Workshops 2016,  pages  740~@~S746,  2016
[2] https://users.ices.utexas.edu/~sreepai/sree-oopsla2016.pdf

INPUT
--------------------------------------------------------------------------------

Input graphs are Galois .csgr format, i.e., symmetric, have no self-loops, and have no duplicated edges.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/gpu/triangle-counting; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm, use the following:

-`$ ./triangle-counting-gpu  <csgr-input-graph>`

-`$ ./triangle-counting-gpu road-USA.csgr`
