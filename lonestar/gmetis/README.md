DESCRIPTION 
===========

This program partitions a given graph using the METIS algorithm:

George Karypis and Vipin Kumar. Multilevel k-way Partitioning Scheme for 
Irregular Graphs. J. Parallel Distributed Computing. 1998.

George Karypis and Vipin Kumar. A fast and high quality multilevel scheme 
for partitioning irregular graphs. International Conference on Parallel 
Processing. 1995

The algorithm first coarsens the graph, partitions it, and then refines 
the partitioning.


INPUT
===========

All algorithm variants expect a directed graph in gr format.


BUILD
===========

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/gmetis; make -j`


RUN
===========

The following are a few example command lines.

-`$ ./gmetis <path-to-graph> <number-of-partitions>`
-`$ ./gmetis <path-to-graph> <number-of-partitions> -t 20 -GGP`


PERFORMANCE
===========

- In our experience, the default GGGP and BKL2 algorithms for initial partitioning 
and refining, respectively, give the best performance.

- The performance of all algorithms depend on an optimal choice of the compile 
time constant, CHUNK_SIZE, the granularity of stolen work when work stealing is 
enabled (via galois::steal()). The optimal value of the constant might depend on 
the architecture, so you might want to evaluate the performance over a range of 
values (say [16-4096]).
