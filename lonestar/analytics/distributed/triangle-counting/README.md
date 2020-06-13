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


PERFORMANCE
--------------------------------------------------------------------------------

The performance analysis of distributed triangle counting can be found at [1]. The key observations from our study are as follows.

* On a single GPU, we do not partition the graph, so application performance is better due to the computation phase on the GPU.

* For distributed multi-GPUs,  we observe that application scales. With the increase in the number of GPUs,  the time taken to compute the number of triangles decreases since our algorithm is free from the synchronization except for the final aggregation.

[1] Loc Hoang, Vishwesh Jatala, Xuhao Chen, Udit Agarwal, Roshan Dathathri, Gurbinder Gill, and Keshav Pingali, [DistTC: High Performance Distributed Triangle Counting. In 2019 IEEE High Performance Extreme Computing Conference](https://ieeexplore.ieee.org/document/8916438), HPEC 2019. IEEE, 2019.
