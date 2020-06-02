## DESCRIPTION

This benchmark computes a minimum spanning tree in a graph. This program uses worklists for better performance.
The algorithm is implemented by successive edge-relaxations of the minimum weight edges. However, since an explicit edge-relaxation involves modifying the graph, the implementation performs edge-relaxation indirectly. This is done by keeping track of the set of nodes that have been merged, called components, which avoids modifications to the graph. Each component's size grows in each iteration, while the number of components reduces (due to components getting merged). 


## BUILD

Run make in the root directory or in the source file (e.g. apps/mst)

## RUN

 ./test path-to-graph -o mst-out
 **NOTE**: Only use the symmetric graph files (.sym.gr) as inputs. 

