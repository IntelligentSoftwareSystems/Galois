Matrix Completion
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------

This benchmark implements Stochastic Gradient Descent (SGD). In particular,
the benchmark uses SGD to complete unknown entries of a sparse matrix.
The sparse matrix represents a bipartite graph, with one set of nodes represent
movies, while the other set represents users. The edge connecting a movie node
to a user node denotes that the user has rated the movie, with the edge label
representing the rating assigned. This benchmark has rough correspondence to
the GPU implementations described
[in this paper](http://www.cs.utexas.edu/~rashid/public/ipdps2016.pdf).

INPUT
--------------------------------------------------------------------------------

This application takes in directed bipartite Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/gpu/matrixcompletion; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm, use the following:

-`$ ./matrixcompletion <input-graph>`
-`$ ./matrixcompletion Epinions_dataset.gr`
