Frequent Subgraph Mining
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This application does frequent subgraph mining in a graph using BFS 
expansion. It uses the bliss library [1][2] for graph isomorphism check.

[1] Bliss: A tool for computing automorphism groups and canonical 
labelings of graphs. http://www.tcs.hut.fi/Software/bliss/, 2017.
[2] Tommi Junttila and Petteri Kaski. 2007. Engineering an efficient 
canonical labeling tool for large and sparse graphs. In Proceedings 
of the Meeting on Algorithm Engineering & Expermiments, 135-149.

INPUT
--------------------------------------------------------------------------------

We support the following input graph formats: **txt**, **adj**.

We mostly use **adj** format as it is also used by Arabesque and RStream.
The **adj** format takes as input graphs with the following formats:

* **Labels on vertices (default)**
```
# <num vertices> <num edges>
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
...
```

We currently do not support graph labels on edges.
Vertex ids are expected to be sequential integers from 0 to (total number of vertices - 1).

This application takes in symmetric and simple graphs.
You must specify both the -symmetricGraph and the -simpleGraph flags when
running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/mining/cpu/frequent-subgraph-mining; make -j`

RUN
--------------------------------------------------------------------------------

The following is an example command line.

-`$ ./frequent-subgraph-mining-cpu <path-to-graph> -symmetricGraph -simpleGraph -k=3 -minsup=300 -t 40`

PERFORMANCE
--------------------------------------------------------------------------------

Please see details in the paper.

