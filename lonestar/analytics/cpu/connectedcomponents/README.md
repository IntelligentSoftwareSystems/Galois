DESCRIPTION 
===========

Find all connected components of an undirected (symmetric) graph. Set the same label
to nodes which belong to the same component.
Two major categories of algorithm: Pointer Jumping and Label Propogation.

Pointer Jumping is based on union-find that nodes' label is a pointer pointing to its
representative. We merge endpoints of edges to form the spanning tree. Merging is done
in two phases to simplify concurrent updates: (1)find components, update the pointer to
reduce the depth of the tree and (2)union componenets, update nodes connected by edges.

In Label Propagation, each node is marked with a unique
label and propagating vertex labels through neighboring vertices until all the vertices in
the same component are labelled with a unique ID.

- Serial: Serial pointer-jumping implementation.
- Synchronous: Bulk synchronous data-driven implementation. Alternatively execute on two 
worklists.
- Async: asynchronous topology-driven implementation. Work unit is a node.
- BlockedAsync: asynchronous topology-driven implementation with NUMA-aware optimization.
Work unit is a node.
- EdgeAsync: asynchronous topology-driven. Work unit is an edge.
- EdgetiledAsync (default): asynchronous topology-driven. Work unit is an edge tile.
- LabelProp: Label propagation implementation.

Pass in a symmetric .sgr graph.

BUILD
===========

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/connectedcomponents; make -j`

RUN
===========

To run default algorithm (edgetiledasync), use the following:
-`$ ./connectedcomponents <input-graph (symmetric)> -t=<num-threads>`

To run a specific algorithm, use the following:
-`$ ./connectedcomponents <input-graph (symmetric)> -t=<num-threads> -algo=<algorithm>'


TUNING PERFORMANCE  
===========
Default algorithm 'edgetiledasync' works best on rmat25, r4-2e26, roadUSA graphs among all algorithms.
Two parameters 'EDGE_TILE_SIZE' and 'CHUNK_SIZE' (granularity of work stealing) are 
crucial to performance and has to be tuned on different platforms. They are set to 
be 512 and 1 respectively by default.
Label propagation is the best if the input graph is randomized, i.e. node ID are randomized,
highest degree node is not node 0.

