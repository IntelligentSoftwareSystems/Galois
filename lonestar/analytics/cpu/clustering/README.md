DESCRIPTION 
===========

This directory contains hierarchical community detection algorithms, that
recursively merge the communities into a single node and perform clustering on the 
coarsened graph until nodes stop changing communities.

2 algorithms are:

* Louvain Clustering: This algorithm uses the modularity function to find well-connected communities
by maximizing the modularity score, which quantifies the quality of node assignments to the 
communities based on the density of connections.
* Leiden Clustering: This is a variant of the Louvain clustering algorithm with the modified
coarsening phase that allows nodes to switch communities even after coarsening. This is 
shown to improve clustering quality with little extra computation.


INPUT
===========

Input is a graph in Galois .sgr format (see top-level README for the project)

BUILD
===========

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/clustering; make -j`


RUN
===========

The following are a few example command lines.

-`$ ./louvainClustering --help` 

-`$ ./louvainClustering <path-to-graph> -t 40 -c_threshold=0.01 -threshold=0.000001 -max_iter 1000 -algo=Foreach  -resolution=0.001`

-`$ ./leidenClustering <path-to-graph> -t 40 -c_threshold=0.01 -threshold=0.000001 -max_iter 1000 -algo=Foreach  -resolution=0.001`
