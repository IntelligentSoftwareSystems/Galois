DESCRIPTION 
===========

This application counts the vertex-induced motifs in a graph using BFS 
expansion. It uses the bliss library [1][2] for graph isomorphism check.

[1] Bliss: A tool for computing automorphism groups and canonical 
labelings of graphs. http://www.tcs.hut.fi/Software/bliss/, 2017.
[2] Tommi Junttila and Petteri Kaski. 2007. Engineering an efficient 
canonical labeling tool for large and sparse graphs. In Proceedings 
of the Meeting on Algorithm Engineering & Expermiments, 135-149.

INPUT
===========

We support four input graph format: **gr**, **txt**, **adj**, **mtx**.

We mostly use **adj** format as also used by Arabesque and RStream.
The **adj** format takes as input graphs with the following formats:

* **Graphs label on vertex(default)**
```
# <num vertices> <num edges>
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
...
```

Vertex ids are expected to be sequential integers between 0 and (total number of vertices - 1).

BUILD
===========

1. Run cmake at BUILD directory `cd build; cmake -DUSE_EXP=1 ../`

2. Run `cd <BUILD>/lonestar/experimental/fsm; make -j`

RUN
===========

The following are a few example command lines.

-`$ ./motif <path-to-graph> -k=3 -t 40`

PERFORMANCE
===========
- I
- I
- I
