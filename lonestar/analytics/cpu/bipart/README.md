Hypergraph Partitioning Decomposition
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Partitiong a hypergraph into <b>k</b> sets. A hypergraph is a generalization of 
graph in which edges can connect more than two nodes. Hypergraph partitioning has 
applications in VLSI, data mining, bioinformatics, and etc. 

The hypergraph is represented as a bipartite graph where one sets of nodes represents
hyperedges and the other set represnts nodes. There is an edge between nodes and a hyperedge
if the node is in that hyperedge.

INPUT
--------------------------------------------------------------------------------

This application takes in **HMetis** inputs .hgr graphs.
You must specify the -hMetisGraph flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/bipart/; make -j

RUN
--------------------------------------------------------------------------------

To run on machine with a k value of 4, use the following:
`./bipart-cpu <input-graph> <number-of-coarsening-levels> <number-of-refinement-levels> -<scheduling-policy> -t=<num-threads> -hMetisGraph`
