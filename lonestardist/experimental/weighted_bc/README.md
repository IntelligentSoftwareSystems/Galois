Betweenness Centrality
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Runs a bulk synchronous parallel version of Brandes's Betweenness Centrality.
The forward and backward phases of Brandes are implemented as a series of 
operators that find shortest distances, calculate the number of predecessors
and successors, calculate the number of shortest paths, and back-propagate
dependency values. The algorithm solves dependencies for a single source source
at a time. Additionally, it uses the weights on the edges of the graph (and,
therefore, is not level by level).

INPUT
--------------------------------------------------------------------------------

Takes in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/dist-apps/; make -j weighted_bc

RUN
--------------------------------------------------------------------------------

To run solving all sources, use the following:
`./weighted_bc <input-graph> -t=<num-threads>`

To run for the first n sources, use the following:
`./weighted_bc <input-graph> -t=<num-threads> -numOfSources=n`

To run on 3 hosts h1, h2, and h3 for the n random sources, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./weighted_bc <input-graph> -t=<num-threads> -numOfSources=n -randomSources`

To run on 3 hosts h1, h2, and h3 with a Cartesian vertex cut partition, use the 
following:
`mpirun -n=3 -hosts=h1,h2,h3 ./weighted_bc <input-graph> -t=<num-threads> -partition=cvc`

PERFORMANCE
--------------------------------------------------------------------------------

This implementation does not perform well on high diameter graphs such as road
networks due to the round based way in which computation occurs.

Additionally, distributing computation for smaller sized graphs will usually
hurt performance: the overhead of synchronization is higher than any 
computational gains that would be achieved by having more computational power.
For larger graphs, distributing computation over multiple hosts does show
scaling.
