Triangle Counting
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Counts the number of triangles in a symmetric, clean (i.e., no self-loops and
no multiedges) graph in a multi-GPU setting. This implementation is the
one used in the paper "DistTC: High Performance Distributed Triangle Counting"
which appeared in the Graph Challenge 2019 competition.

A CPU implementation is currently in planning and will appear here once it is
ready.

INPUT
--------------------------------------------------------------------------------

Takes in symmetric Galois .gr graphs that have been cleaned.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).
The DIST HETERO option to enable GPUs must be on. (see README in lonestardist)

2. Run `cd <BUILD>/lonestardist/tc; make -j

RUN
--------------------------------------------------------------------------------

To run on 1 with a single GPU, use the following:
`./tc <symmetric-input-graph> -pset=g -num_nodes=1`

To run on 3 GPUs on a machine, use the following:
`mpirun -n=3 ./tc <symmetric-input-graph> -pset=ggg -num_nodes=1`

To run on 6 GPUs on 2 machines h1 and h2 with 3 GPUs each, use the following:
`mpirun -n=6 -hosts=h1,h2 ./tc <symmetric-input-graph> -pset=ggg -num_nodes=2`

PERFORMANCE
--------------------------------------------------------------------------------

Uneven load balancing among hosts can heavily affect performance.
