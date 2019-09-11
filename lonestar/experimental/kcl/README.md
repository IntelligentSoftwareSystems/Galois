DESCRIPTION 
===========

This application counts the k-Cliques in a graph 
kcl: BFS vertex extension using SoA embedding list.
kcl_queue: BFS vertex extension using AoS embedding queue.
kcl_dfs: DFS vertex extension.

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

2. Run `cd <BUILD>/lonestar/experimental/kcl; make -j`

RUN
===========

The following are a few example command lines.

-`$ ./kcl <path-to-graph> -k=3 -t 40`

PERFORMANCE
===========
- I
- I
- I
