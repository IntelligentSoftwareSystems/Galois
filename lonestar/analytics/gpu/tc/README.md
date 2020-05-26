## DESCRIPTION

This benchmark counts the number of triangles in a given undirected graph. It implements the approach from Polak [1] in IrGL[2].

[1] Adam Polak. Counting triangles in large graphs on GPU. In IPDPS Workshops 2016,  pages  740–746,  2016
[2] https://users.ices.utexas.edu/~sreepai/sree-oopsla2016.pdf

## BUILD

Assuming CMake is performed in the ${GALOIS\_ROOT}/build, compile the application by executing the
following command in the ${GALOIS\_ROOT}/build/lonestar/analytics/gpu/triangle directory.

`make -j`

## RUN

Execute as: ./triangle-counting [-o output-file] undirected-graph-file

e.g., ./triangle-counting -o outfile.txt road-USA.sgr
