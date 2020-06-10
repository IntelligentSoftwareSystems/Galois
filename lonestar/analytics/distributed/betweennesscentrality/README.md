Betweenness Centrality
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

betweenesscentrality-level is a bulk synchronous parallel version of Brandes's
Betweenness Centrality that does both the forward and backward phases of
Brandes's algorithm in a level by level, work-efficient fashion. The algorithm
solves dependencies for a single source at a time.

betweenesscentrality-minrounds is a provably round efficient distributed
algorithm that can solve for betweenness centrality dependencies for multiple
sources at a time. It leverages a proven insight that allows the algorithm to
know exactly which round that synchronization of source data needs to occur:
this results in communication only when necessary which further improves the
algorithms efficiency in the distributed setting. Details of the algorithm, the
proofs of correctness, and performance comparisons can be found in our paper:

Loc Hoang, Matteo Pontecorvi, Roshan Dathathri, Gurbinder Gill, Bozhi You,
Keshav Pingali, Vijaya Ramachandran, “A Round-Efficient Distributed
Betweenness Centrality Algorithm,” Proceedings of the 24th ACM SIGPLAN
Symposium on Principles and Practice of Parallel Programming (PPoPP), February
2019.


INPUT
--------------------------------------------------------------------------------

Takes in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/distributed/betweenesscentrality; make -j`

RUN
--------------------------------------------------------------------------------

All the command line arguments used by both apps are the same except for
`-numRoundSources`, which is used by minrounds to control the number of sources
being batched at any given point.

To run solving for all sources, use the following:
`./betweenesscentrality-level-dist <input-graph> -t=<num-threads>`

To run for the first n sources, use the following:
`./betweenesscentrality-level-dist <input-graph> -t=<num-threads> -numOfSources=n`

To run using specified sources from a file, use the following:
`./betweenesscentrality-level-dist <input-graph> -t=<num-threads> -sourcesToUse=<filename>`

To run on 3 hosts h1, h2, and h3 with a Cartesian vertex cut partition for all
sources, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./betweenesscentrality-level-dist <input-graph> -t=<num-threads> -partition=cvc`

To run for all sources in batches of k on 3 hosts, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./betweenesscentrality-minrounds-dist <input-graph> -t=<num-threads> -numRoundSources=k`

PERFORMANCE
--------------------------------------------------------------------------------

* The minrounds implementation performs significantly better than the level
implementation on high diameter graphs as it batches multiple sources together
at once and significantly reduces (1) rounds executed and (2) the communication
overhead. 

* Batching more sources in minrounds is a tradeoff between memory usage
and efficiency: more sources generally leads to less rounds executed but
requires a linear increase in memory used by the implementation to store data
for all of the sources being batched.

* More details on the differences between level and minrounds can be found in
our performance study in the MRBC paper cited above.
