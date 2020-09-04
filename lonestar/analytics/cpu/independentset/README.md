Maximal Independent Set
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Find the Maximal Independent Set (not maximum) of ndoes in an undirected 
(symmetric) graph. 

For convenience, we used IN to represent a node in the independent set, OUT to 
represent not in the independent set, UNDECIDED represent undecided.

- serial: serial greedy version.
- pull: pull-based greedy version. Node 0 is initially marked IN.
- detBase: greedy version, using Galois deterministic worklist.
- nondet: greedy version, using Galois bulk synchronous worklist.
- prio(default): based on Martin Butcher's GPU ECL-MIS algorithm. For more information,
  please look at http://cs.txstate.edu/~burtscher/research/ECL-MIS/.
- edgetiledprio: edge-tiled version of prio.

INPUT
--------------------------------------------------------------------------------

This application takes in symmetric Galois .gr graphs.
You must specify the -symmetricGraph flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/independentset/; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm (prio), use the following:
-`$ ./maximal-independentset-cpu <input-graph (symmetric)> -t=<num-threads> -symmetricGraph`

To run a specific algorithm, use the following:
-`$ ./maximal-independentset-cpu <input-graph (symmetric)> -t=<num-threads> -algo=<algorithm> -symmetricGraph`

PERFORMANCE  
--------------------------------------------------------------------------------

In 'prio', when a node has high priority than all of its neighbors, it is marked 
as IN. For its neighbors, you can choose either 1) update its neighbors to OUT in 
same round (Push), or 2) next round its neighbors check if they have an IN neighbor, 
and update themselves to OUT (Pull).
First method works better on none-power-law graphs. Second method works better 
on power-law graphs. 
