## DESCRIPTION

A connected component of an undirected graph is a subgraph in which there is a path between any two nodes. A node with no edges is itself a connected component. This benchmark computes number of connected components in an undirected graph.

## BUILD

Assuming CMake is performed in the ${GALOIS\_ROOT}/build, compile the application by executing the
following command in the ${GALOIS\_ROOT}/build/lonestar/analytics/gpu/connected-components directory.

`make -j`

## RUN

Execute as: ./connected-components [-o output-file] undirected-graph-file

e.g., ./connected-components -o outfile.txt road-USA.sgr
