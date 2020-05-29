## DESCRIPTION
This benchmark implements Stochastic Gradient Descent (SGD). In particular, the benchmark uses SGD to complete unknown entries of a sparse matrix.
The sparse matrix represents a bipartite graph, with one set of nodes represent movies, while the other set represents users.
The edge connecting a movie node to a user node denotes that the user has rated the movie, with the edge label representing the rating assigned.
This benchmark has rough correspondence to the GPU implementations described [in this paper](http://www.cs.utexas.edu/~rashid/public/ipdps2016.pdf).

## COMPILE

Simply run make in the root directory or in the source code directory (e.g. apps/sgd)

## Run

./blk_diag path-to-input


