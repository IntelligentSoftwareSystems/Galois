README related to SHAD input graph ingestion
(Including some notes for other workflows)
This README is for our internal purpose.
This README will be refined with more concrete information later.

1. CMakeList paths:
The current CMake in Galois is using hard-:coded paths for CUDA_HOME,
OPENBLAS_ROOT, INTEL_COMPILER_LIBRARIES, and MKL_LIBRARIES.
Please set those variables based on your environments.


2. Assumptions regarding SHAD WMD graph formats:
We assume that in SHAD WMD graph formats, each node and edge has a single type,
and those types are ALWAYS uint64_t.
The current Galois does not support node/edge properties (possibly,
programmers can implement a struct containing multiple
fields, but that is not like getData<Property1>(n), getData<Property2>(n), etc.)
and so, we store those SHAD types in node and edge data.
If you need other types than uint64_t, you should add new execution paths for
them.


3. Limitations of the current SHAD graph ingestion module:
In the original CuSP, each host reads parts of the .gr graph file and constructs
in-memory format. In this case, each host does not need to load the full graph
in its memory space. This is possible since .gr file is CSR and each component
such as outgoing edge indices, outgoing edge destinations, and outgoing edge
data is stored consecutively.

However, in the SHAD graph format, all components are not stored consecutively.
They are unsorted. For example, edges and nodes can be stored in interleaved
manner. Therefore, it is not possible to read partial graphs by using
the original method. 

As the current SHAD graph ingestion does not focus on decent/scalable methods,
but to make SHAD graphs work in Galois to proceed with workflows,
each host reads the FULL graph to in-memory. This should NOT be the final
artifact since our long-run target graphs should exceed a single machine memory.
But for the immediate goal and the target data sets, I assume that it is fine
for now.

UT team is currently working on new graph formats for dynamic graphs, and 
scalable SHAD graph ingestion across hosts.

4. TODO:
CuSP marks training/test/validation nodes while it is partitioning a graph.
It is not implemented yet for a SHAD graph.
This will be added in a GNN/feature construction branch.

5. Requirements:
Galois-GNN requires additional packages listed below on top of the requirements of Galois.
You can use older/newer versions but let me (hochan) also list the versions that I have used:
1) Intel MKL: 2023.1.0
2) Intel Compiler (including runtime libraries): 2023.0.0
3) Intel Onedpl-devel library: 2023.1.0
4) Intel OpenMP: 2023.0.0

