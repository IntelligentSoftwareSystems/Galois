Connected Components (Pull)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Find all connected components of an undirected (symmetric) graph. Set the same 
label to nodes which belong to the same component.

The algorithm is a bulk synchronous parallel label propagating algorithm.
In this pull variant of the algorithm, all nodes check their neighbors
to see if they have a lower label, and they will adopt the lowest label
among its neighbors/itself as its component.

INPUT
--------------------------------------------------------------------------------

Takes in symmetric Galois .gr graphs. You must specify the -symmetricGraph
flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/dist-apps/; make -j cc_pull

RUN
--------------------------------------------------------------------------------

To run on 1 machine, use the following:
`./cc_pull <symmetric-input-graph> -t=<num-threads> -symmetricGraph`

To run on 3 hosts h1, h2, and h3, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./cc_pull <symmetric-input-graph> -t=<num-threads> -symmetricGraph`

To run on 3 hosts h1, h2, and h3 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./cc_pull <symmetric-input-graph> -t=<num-threads> -symmetricGraph -partition=iec`

PERFORMANCE  
--------------------------------------------------------------------------------

In our experience, the pull variant of cc does not perform as well as the push
variant.

Uneven load balancing among hosts can affect performance.
