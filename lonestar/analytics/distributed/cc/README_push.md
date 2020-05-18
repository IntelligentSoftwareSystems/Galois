Connected Components (Push)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Find all connected components of an undirected (symmetric) graph. Set the same 
label to nodes which belong to the same component.

The algorithm is a bulk synchronous parallel label propagating algorithm.
In this push variant of the algorithm, nodes with a label that has changed
from the last round will push this label out to its neighbors and update their
labels with a min operation.

INPUT
--------------------------------------------------------------------------------

Takes in symmetric Galois .gr graphs. You must specify the -symmetricGraph
flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/dist-apps/; make -j cc_push

RUN
--------------------------------------------------------------------------------

To run on 1 machine, use the following:
`./cc_push <symmetric-input-graph> -t=<num-threads> -symmetricGraph`

To run on 3 hosts h1, h2, and h3, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./cc_push <symmetric-input-graph> -t=<num-threads> -symmetricGraph`

To run on 3 hosts h1, h2, and h3 with an incoming edge cut, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./cc_push <symmetric-input-graph> -t=<num-threads> -symmetricGraph -partition=iec`

PERFORMANCE  
--------------------------------------------------------------------------------

Uneven load balancing among hosts can affect performance.
