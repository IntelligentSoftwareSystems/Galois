Overview of Graph Pattern Mining (GPM) in Galois
================================================================================

This is the Pangolin framework [1] for efficient and flexible 
graph pattern mining based on the Galois programming system. 
It uses the bliss library [2][3] for graph isomorphism check. 
The license for this library is in the bliss directory: 
note that **it does not use the same license as the rest of Galois**.

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali, 
Pangolin: An Efficient and Flexible Graph Pattern Mining System on CPU and GPU, arXiv:1911.06969
[2] Bliss: A tool for computing automorphism groups and canonical 
labelings of graphs. http://www.tcs.hut.fi/Software/bliss/, 2017.
[3] Tommi Junttila and Petteri Kaski. 2007. Engineering an efficient 
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

We currently do not support graphs label on edges

Vertex ids are expected to be sequential integers between 0 and (total number of vertices - 1).

BUILD
===========

1. Run cmake at BUILD directory `cd build; cmake -DUSE_PANGOLIN=1 ../`

2. Run `cd <BUILD>/lonestar/experimental/fsm; make -j`

RUN
===========

The following are a few example command lines.

-`$ ./tc <path-to-graph> -t 40`
-`$ ./kcl <path-to-graph> -k=3 -t 40`
-`$ ./motif <path-to-graph> -k=3 -t 40`
-`$ ./fsm <path-to-graph> -k=3 -minsup=300 -t 40`

PERFORMANCE
===========
- I
- I
- I
