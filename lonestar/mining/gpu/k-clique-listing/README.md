K-Clique
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This application counts the K-Cliques in a graph.

INPUT
--------------------------------------------------------------------------------

This application takes in symmetric and simple Galois .gr graphs.
You must specify both the -symmetricGraph and the -simpleGraph flags when
running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/mining/gpu/k-clique-listing; make -j`

RUN
--------------------------------------------------------------------------------

The following is an example command line.

-`$ ./k-clique-listing-gpu -symmetricGraph -simpleGraph <path-to-graph> -k=3 -t 40`

PERFORMANCE
--------------------------------------------------------------------------------

Please see details in the paper.
