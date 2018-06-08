DESCRIPTION 
===========

This program finds the maximum cardinality bipartite matching in a bipartite graph.
It uses the Alt-Blum-Melhorn-Paul Algorithm described at
https://web.eecs.umich.edu/~pettie/matching/Alt-Blum-Mehlhorn-Paul-bipartite-matching-dense-graphs.pdf
This algoritm is also described in:
K. Mehlhorn and S. Naeher. LEDA: A Platform for Combinatorial and Geometric Computing. Cambridge University Press, 1999

After all the augmenting paths of a given length are found, the algorithm finishes using the Ford-Fulkerson algorithm for matching.

By default, a randomly generated input is used, though input can be taken from a file instead.
In general, the parallelism available to this algorithm is heavily dependent on the characteristics of the input.

BUILD
=====

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/matching && make -j`


RUN
===

The following are a few example command lines.

 - `./bipartite-mcm -abmpAlgo -inputType=generated -numEdges=100000000 -numGroups=10000 -seed=0 -n=1000000 -t=40`
 - `./bipartite-mcm -abmpAlgo -inputType=generated -numEdges=1000000000 -numGroups=2000000 -seed=0 -n=10000000 -t=40`

