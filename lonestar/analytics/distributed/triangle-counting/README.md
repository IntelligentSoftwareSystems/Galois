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
You must specify the -symmetricGraph flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/distributed/triangle-counting; make -j

RUN
--------------------------------------------------------------------------------

To run on 1 with a single GPU, use the following:
`./triangle-counting-dist <symmetric-input-graph> -symmetricGraph -pset=g -num_nodes=1`

To run on a single machine with 56 CPU threads, use the following:
`./triangle-counting-dist <symmetric-input-graph> -symmetricGraph -t=56`

To run on 3 GPUs on a machine, use the following:
`mpirun -n=3 ./triangle-counting-dist <symmetric-input-graph> -symmetricGraph -pset=ggg -num_nodes=1`

To run on 6 GPUs on 2 machines h1 and h2 with 3 GPUs each, use the following:
`mpirun -n=6 -hosts=h1,h2 ./triangle-counting-dist <symmetric-input-graph> -symmetricGraph -pset=ggg -num_nodes=2`

To run on 4 GPUs and 2 CPUs on 2 machines h1 and h2 with 2 GPUs and 1 CPU each, use the following:
`mpirun -n=6 -hosts=h1,h2 ./triangle-counting-dist <symmetric-input-graph> -symmetricGraph -pset=ggc -num_nodes=2`
