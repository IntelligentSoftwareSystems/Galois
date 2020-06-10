Triangle Counting
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program counts the number of triangles in a given undirected graph.

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

Vertex ids are expected to be sequential integers between 0 and (total number of vertices - 1).

This application takes in symmetric and simple Galois .gr graphs.
You must specify both the -symmetricGraph and the -simpleGraph flags when
running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/mining/gpu/triangle-counting; make -j`

RUN
--------------------------------------------------------------------------------

The following is an example command line.

-`$ ./triangle-counting-gpu -symmetricGraph -simpleGraph <path-to-graph> -k=3 -t 40`
