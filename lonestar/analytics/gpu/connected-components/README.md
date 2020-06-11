Connected Components
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------


A connected component of an undirected graph is a subgraph in which there is a path between any two nodes. A node with no edges is itself a connected component. This benchmark computes number of connected components in an undirected graph.

INPUT
--------------------------------------------------------------------------------

Take in symmetric Galois .sgr graphs. 

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/gpu/connected-components; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm, use the following:

-`$ ./connected-components-gpu -o <output-file> <symmetric-input-graph>`

-`$ ./connected-components-gpu -o outfile.txt road-USA.sgr`

