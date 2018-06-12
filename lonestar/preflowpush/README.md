DESCRIPTION 
===========

This program computes the maximum flow from a given source to a given sink 
in a given graph using the preflow-push algorithm (also called push-relabel 
algorithm). Please refer to the textbook for more details on the algorithm:

Cormen, Leiserson, Rivest, Stein. Introduction to Algorithms. MIT Press. 2001.


INPUT
===========

All algorithm variants expect a directed graph in gr format.


BUILD
===========

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/preflowpush; make -j`


RUN
===========

The following are a few example command lines.

-`$ ./preflowpush <path-to-graph> <source-ID> <sink-ID>`
-`$ ./preflowpush <path-to-graph> <source-ID> <sink-ID> -t=20`


PERFORMANCE
===========

- In our experience, the deterministic algorithms perform much slower than the 
non-deterministic one.

- The performance of all algorithms depend on an optimal choice of the compile 
time constant, CHUNK_SIZE, the granularity of stolen work when work stealing is 
enabled (via galois::steal()). The optimal value of the constant might depend on 
the architecture, so you might want to evaluate the performance over a range of 
values (say [16-4096]).
