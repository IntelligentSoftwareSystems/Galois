DESCRIPTION 
===========

We implement both pull and push-style PageRank algorithms. The push-style
algorithms are based on the computations (Algorithm 4) described in the 
PageRank Europar 2015 paper.

Whang et al. Scalable Data-driven PageRank: Algorithms, System Issues, and 
Lessons Learned. Europar 2015.

There are two variants, topological and residual, of the pull-style algorithm 
that are implemented. The pull variants perform better than the push variants 
since there are no atomic operations. The residual version performs and scales 
the best. It does less work and uses separate arrays for storing delta and 
residual information to improve locality and use of memory bandwidth.


INPUT
===========

For the push variant, input is a graph in Galois .gr format (see top-level 
README for the project). Note that the pull variants expect a transpose graph. 
For the pull variant, input is a graph is Galois .tgr format. 


BUILD
===========

1. Run `cmake` at the BUILD directory (refer to top-level README for instructions).

2. Run `cd <BUILD>/lonestar/pagerank; make -j` 


RUN
===========

The following are a few examples of invoking PageRank.

* `$ ./pagerank-pull <path-transpose-graph> -tolerance=0.001`

* `$ ./pagerank-pull <path-transpose-graph> -t=20 -tolerance=0.001 -algo=Residual`

* `$ ./pagerank-push <path-graph> -t=40 -tolerance=0.001 -algo=Async`


TUNING PERFORMANCE  
===========

The performance of the push and the pull versions depend on an optimal choice 
of the the compile time constant, CHUNK_SIZE. For the pull version, CHUNK_SIZE 
denotes the granularity of stolen work when work stealing is enabled (via 
galois::steal()). The optimal value of the constant might depend on the 
architecture, so you might want to evaluate the performance over a range of 
values (say [16-4096]).
