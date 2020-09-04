Overview of Graph Pattern Mining (GPM) in Galois
================================================================================

This directory contains benchmarks for efficient and flexible graph mining 
that run using the Pangolin framework [1] on a multi-core CPU or a GPU. 
It uses the bliss [2][3] library v0.73 for graph isomorphism test. 
The license for this library is in the bliss
directory: note that **it does not use the same license as the rest of Galois**.  

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali, 
Pangolin: An Efficient and Flexible Graph Pattern Mining System on CPU and GPU, VLDB 2020

[2] Bliss: A tool for computing automorphism groups and canonical 
labelings of graphs. http://www.tcs.hut.fi/Software/bliss/, 2017.

[3] Tommi Junttila and Petteri Kaski. 2007. Engineering an efficient 
canonical labeling tool for large and sparse graphs. In Proceedings 
of the Meeting on Algorithm Engineering & Expermiments, 135-149.

Compiling GPM Applications Through CMake 
================================================================================

The dependencies for LonestarGPU suite are the same as shared-memory.
Note that  LonestarGPU requires CUDA 8.0 and above.

Note that heterogeneous Galois requires the cub and moderngpu git submodules,
which can be cloned using the followed commands.

```Shell
cd $GALOIS_ROOT
git submodule init
git submodule update
```
These modules will be cloned in the ${GALOIS\_ROOT}/external directory

Mining applications for CPU are enabled by default.
To build the mining applications for GPU, first, create a build directory and
run CMake with -DGALOIS\_CUDA\_CAPABILITY=\<insert CUDA capability here\> flag
in the build directory. The CUDA capability should be one that your
GPU supports. For example, if you wanted to build for a GTX 1080 and a K80,
the commands would look like this:

```Shell
cd ${GALOIS_ROOT}
mkdir build
cd build
cmake ${GALOIS_ROOT} -DGALOIS_CUDA_CAPABILITY="3.7;6.1"
```

After compiling through CMake, the system will create the 'lonestar/mining/cpu' 
and 'lonestar/mining/gpu' directories in ${GALOIS\_ROOT}/build directory. 

Compiling Mining Applications
================================================================================

Once CMake is completed,  compile the provided mining apps by executing the 
following command in the ${GALOIS\_ROOT}/build/lonestar/mining directory.

```Shell
`make -j`
```

You can compile a specific app by executing the following commands (shown for motif-counting on CPU).

```Shell
cd cpu/motif-counting
make -j
```

INPUT
================================================================================

We support four input graph format: **gr**, **txt**, **adj**, **mtx**.
For unlabeled graphs, we use the gr graph format, same as other Galois benchmarks.
**Make sure that the graph is symmetric and contains no self-loop or redundant edges**.
If not, use the convert tool in tools/graph-convert/ to convert the graph.
We use **adj** format for labeled graphs as also used by Arabesque and RStream.
The **adj** format takes as input graphs with the following formats (vertex labeled):

```
# <num vertices> <num edges>
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
...
```

We currently do not support graphs label on edges.

Vertex ids are expected to be sequential integers between 0 and (total number of vertices - 1).
For testing, we have prepared a test graph **citeseer**. After running make input,
the needed input files can be build in "$BUILD_DIR/inputs/Mining".

Running Provided Apps
================================================================================

The following are a few example command lines.

- `$ ./triangle-counting-mining-cpu -symmetricGraph -simpleGraph $BUILD_DIR/inputs/Mining/citeseer.csgr -t 28`
- `$ ./k-clique-listing-cpu -symmetricGraph -simpleGraph $BUILD_DIR/inputs/Mining/citeseer.csgr -k=3 -t 28`
- `$ ./motif-counting-cpu -symmetricGraph -simpleGraph $BUILD_DIR/inputs/Mining/citeseer.csgr -k=3 -t 56`
- `$ ./frequent-subgraph-mining-cpu -symmetricGraph -simpleGraph $BUILD_DIR/inputs/Mining/citeseer.sadj -ft adj -k=2 -ms=300 -t 28`

PERFORMANCE
================================================================================

Please see details in the paper.

CITATION
================================================================================

Please cite the following paper if you use Pangolin:

```
@article{Pangolin,
	title={Pangolin: An Efficient and Flexible Graph Mining System on CPU and GPU},
	author={Xuhao Chen and Roshan Dathathri and Gurbinder Gill and Keshav Pingali},
	year={2020},
	journal = {Proc. VLDB Endow.},
	issue_date = {August 2020},
	volume = {13},
	number = {8},
	month = aug,
	year = {2020},
	numpages = {12},
	publisher = {VLDB Endowment},
}
```
