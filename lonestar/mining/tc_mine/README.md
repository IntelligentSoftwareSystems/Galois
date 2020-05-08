DESCRIPTION 
===========

This application counts the triangles in the input graph.
tc: extension from edge list, DAG enabled.
tc_vertex: extension from each vertex, DAG enabled.
tc_naive: naive implementation using undirected graph.

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

2. Run `cd <BUILD>/lonestar/experimental/tc; make -j`

RUN
===========

The following are a few example command lines.

-`$ ./tc <path-to-graph> -k=3 -t 40`

PERFORMANCE
===========
- I
- I
- I
