Matrix Completion
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Runs matrix completion using stochastic gradient descent.

The algorithm is a bulk synchronous parallel residual based algorithm. In
each round, updates to the latent vectors are calcuated based on the current
error between 2 nodes and then applied at the end of the round.

INPUT
--------------------------------------------------------------------------------

Takes in bipartite Galois .gr graphs: all nodes with edges should be located
in the prefix of the graph.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/distributed/matrixcompletion; make -j

RUN
--------------------------------------------------------------------------------

To run for a max of 10 iterations, do the following
`./matrixcompletion-dist <bipartite-input-graph> -t=<num-threads> -maxIterations=10`

To run on 3 hosts h1, h2, and h3 with changes to the learning parameters, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./matrixcompletion-dist <bipartite-input-graph> -t=<num-threads> -DECAY_RATE=0.5 -LAMBDA=0.001 -LEARNING_RATE=0.001`

PERFORMANCE  
--------------------------------------------------------------------------------

* Convergence/time to convergence may be affected by the different learning 
  parameters (e.g. decay rate, lambda, learning rate). They may need tuning for
  best performance. The best parameters are input dependent.

* For 16 or less hosts/GPUs, for performance, we recommend using an
  **edge-cut** partitioning policy (OEC or IEC).

* For 32 or more hosts/GPUs, for performance, we recommend using the
  **Cartesian vertex-cut** partitioning policy (CVC).
