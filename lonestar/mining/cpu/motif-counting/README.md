Motif Counting
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This application counts the motifs in a graph using BFS 
expansion. It uses the bliss library [1][2] for graph isomorphism test.

[1] Bliss: A tool for computing automorphism groups and canonical 
labelings of graphs. http://www.tcs.hut.fi/Software/bliss/, 2017.
[2] Tommi Junttila and Petteri Kaski. 2007. Engineering an efficient 
canonical labeling tool for large and sparse graphs. In Proceedings 
of the Meeting on Algorithm Engineering & Expermiments, 135-149.

INPUT
--------------------------------------------------------------------------------

We support the following four input graph formats: **gr**, **txt**, **adj**, **mtx**.

We mostly use **adj** format as also used by Arabesque and RStream.
The **adj** format takes as input graphs with the following formats:

* **Graphs label on vertex(default)**
```
# <num vertices> <num edges>
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
...
```

Vertex ids are expected to be sequential integers from 0 to (total number of vertices - 1).

This application takes in symmetric and simple Galois .gr graphs.
You must specify both the -symmetricGraph and the -simpleGraph flags when
running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/mining/cpu/motif-counting; make -j`

RUN
--------------------------------------------------------------------------------

The following is an example command line.

-`$ ./motif-counting-cpu -symmetricGraph -simpleGraph <path-to-graph> -k=3 -t 28`
