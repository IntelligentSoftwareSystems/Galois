Betweenness Centrality (Outer)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Runs Betweenness Centrality where the unit of parallelism is a betweenness 
centrality source. Each thread will work on the betweenness centrality
computation of it own individual source and find the BC contributions of that
source to the rest of the graph.

Pass in a regular .gr graph.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/betweennesscentrality; make -j betweennesscentrality-outer`

RUN
--------------------------------------------------------------------------------

To run all sources, use the following:
`./betweennesscentrality-outer <input-graph> -t=<num-threads>

To run starting from a particular source, use the following:
`./betweennesscentrality-outer <input-graph> -t=<num-threads> -startNode=<node to begin>`

To run only on N nodes (that have outgoing edges), use the following:
`./betweennesscentrality-outer <input-graph> -t=<num-threads> -limit=N`

TUNING PERFORMANCE  
--------------------------------------------------------------------------------

If each source's BC calculation takes roughly the same amount of time, then
load balancing should be good. Otherwise, there may be load imbalance among 
threads.


Asynchronous Brandes Betweenness Centrality
================================================================================

DESCRIPTION 
----------------------------------------

Runs an asynchronous version of Brandes's Betweenness Centrality as formulated
through the operator formulation of algorithms. It is a two-phase algorithm:
the first phase determines the shortest path DAG and counts the number of 
shortest paths through a given node for a particular source, and the second
phase back-propagates dependency values for the calculation of betweenness
centrality.

control.h has some variables that may alter how the algorithm runs and what kind
of data it collects.

Pass in a regular .gr graph.

For more details on the algorithm, see paper here:
https://dl.acm.org/citation.cfm?id=2442521

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/betweennesscentrality; make -j bc-async`

RUN
--------------------------------------------------------------------------------

To run all sources, use the following:
`./bc-async <input-graph> -t=<num-threads>`

To run with a specific number of sources N (starting from the beginning), use
the following:
`./bc-async <input-graph> -t=<num-threads> -numOfSources=N`

To run with a specific number of sources N (starting from the beginning) **with
outgoing edges**, use the following:
`./bc-async <input-graph> -t=<num-threads> -numOfOutSources=N`

To run with a specific set of sources, put the sources in a file with
the source ids separated with a line and use the following:
`./bc-async <input-graph> -t=<num-threads> -sourcesToUse=<path-to-file>`

TUNING PERFORMANCE  
--------------------------------------------------------------------------------

Good scaling and performance is very dependent on the chunk size parameter
for the worklist. It must be changed through the source code as it is
a compile time variable used in templates. The best chunk size is input
dependent.

Good scaling also comes from using the Galois power-of-two allocator
for memory allocations in parallel regions.

Finally, it may be useful to toggle BC_USE_MARKING in control.h: if on, it will
check to see if a node is in a worklist before adding it (preventing duplicates).
Depending on the input graph, performance may improve with this setting on.
