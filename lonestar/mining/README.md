Overview of Graph Pattern Mining (GPM) in Galois
================================================================================

This directory contains benchmarks that run using the Pangolin framework [1]
for efficient and flexible graph mining. It uses the bliss [2][3] library v0.73
for graph isomorphism test. The license for this library is in the bliss
directory: note that **it does not use the same license as the rest of Galois**.  

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali, 
Pangolin: An Efficient and Flexible Graph Pattern Mining System on CPU and GPU, VLDB 2020

[2] Bliss: A tool for computing automorphism groups and canonical 
labelings of graphs. http://www.tcs.hut.fi/Software/bliss/, 2017.

[3] Tommi Junttila and Petteri Kaski. 2007. Engineering an efficient 
canonical labeling tool for large and sparse graphs. In Proceedings 
of the Meeting on Algorithm Engineering & Expermiments, 135-149.

Compiling Provided Apps
================================================================================

Pangolin built by default. To enable GPU mining, you can give the
`-DGALOIS_ENABLE_GPU=ON` setting to `cmake`.

Note that heterogeneous Galois requires the cub git submodules, which can be cloned using the followed commands.

```Shell
cd $GALOIS_ROOT
git submodule init
git submodule update --remote
```
These modules will be cloned in the ${GALOIS\_ROOT}/external directory

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
For testing, we have prepared a test graph **citeseer** in `$SRC_DIR/lonestarmine/test_data`.

Running Provided Apps
================================================================================

The following are a few example command lines.

- `$ ./tc_mine $SRC_DIR/lonestarmine/test_data/citeseer.csgr -t 28`
- `$ ./kcl $SRC_DIR/lonestarmine/test_data/citeseer.csgr -k=3 -t 28`
- `$ ./motif $SRC_DIR/lonestarmine/test_data/citeseer.csgr -k=3 -t 56`
- `$ ./fsm $SRC_DIR/lonestarmine/test_data/citeseer.sadj -ft adj -k=2 -ms=300 -t 28`

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
