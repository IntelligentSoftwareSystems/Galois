DESCRIPTION

This program performs the matrix completion using different stochastic gradient descent (SGD) and alternating least squares (ALS) algorithms on a bipartite graph.
We have implemeted 4 SGD based algorithms and 2 ALS based algorithms.

SGD algorithms:
1. sgdByItems
2. sgdByEdges
3. sgdBlockEdge
4. sgdBlockJump

ALS algorithms:
1. SimpleALS
2. SyncALS

All versions expect a bipartite graph in gr format.
NOTE: The bipartite must have all the nodes with out-going edges in the beginning, followed by all the nodes without any out-going edges.
For example, a bipartite graph with out-going edges from users to movies, where each edge is a rating given by a user for a movie, the graph
layout must have all the user nodes together in the beginning followed by all the movie nodes.


BUILD

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/matrixcompletion; make -j`


RUN

The following are a few example command lines.

`$./matrixCompletion <path-symmetric-graph> -algo=sgdBlockJump  -lambda=0.001 -learningRate=0.01 -learningRateFunction=intel -tolerance=0.0001 -t 40 -updatesPerEdge=1 -maxUpdates=20`

To list all the options including the names of the algorithms (-algo):
`$./matrixCompletion --help`

In our experience, out of all the SGD algorithms on netflix graph (#nodes: 497959, #edges: 99072112), sgdBlockEdge
gives the best performance and out of ALS algorithms SyncALS performs the best.



TUNING PERFORMANCE

Performance of different algorithmic variants is input dependent. 
The values for '-lambda', '-learningRateFunction', and '-learningRate' need 
to be tuned for each input graph. If root mean square erro (RMSE) is 'nan', try 
different values for 'lambda', 'learningRateFunction', and 'learningRate'.
