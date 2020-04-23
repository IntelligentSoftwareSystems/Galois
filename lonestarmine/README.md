Overview of Graph Pattern Mining (GPM) in Galois
================================================================================

This directory contains benchmarks that run using the Pangolin library[1].

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali, 
Pangolin: An Efficient and Flexible Graph Mining System on CPU and GPU, VLDB 2020

BUILD
===========

1. Run cmake at BUILD directory `cd build; cmake -DUSE_PANGOLIN=1 ../`
To enable GPU mining, do `cmake -DUSE_PANGOLIN=1 -DUSE_GPU=1 ../`

2. Run `cd <BUILD>/lonestar/experimental/fsm; make -j`

INPUT
===========

We support four input graph format: **gr**, **txt**, **adj**, **mtx**.
For unlabeled graphs, we use the gr graph format, same as other Galois benchmarks.
**Make sure that the graph is symmetric and contains no self-loop or redundant edges**.
If not, use the convert tool in tools/graph-convert/ to convert the graph.
We use **adj** format for labeled graphs as also used by Arabesque and RStream.
The **adj** format takes as input graphs with the following formats:

* **Graphs label on vertex(default)**
```
# <num vertices> <num edges>
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
<vertex id> <vertex label> [<neighbour id1> <neighbour id2> ... <neighbour id n>]
...
```

We currently do not support graphs label on edges

Vertex ids are expected to be sequential integers between 0 and (total number of vertices - 1).

For testing, we have prepared a test graph **citeseer** in $GALOIS_HOME/lonestarmine/test_data/

RUN
===========

The following are a few example command lines.

- `$ ./tc_mine gr $GALOIS_HOME/lonestarmine/test_data/citeseer.csgr -t 28`
- `$ ./kcl gr $GALOIS_HOME/lonestarmine/test_data/citeseer.csgr -k=3 -t 28`
- `$ ./motif gr $GALOIS_HOME/lonestarmine/test_data/citeseer.csgr -k=3 -t 56`
- `$ ./fsm adj $GALOIS_HOME/lonestarmine/test_data/citeseer.sadj -k=2 -ms=300 -t 28`

PERFORMANCE
===========

- Please see details in the paper

CITATION
==========

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

