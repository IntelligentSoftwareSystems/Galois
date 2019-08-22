k-Clique
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Find all size k cliques in an undirected (symmetric) graph. 

The algorithm is a BFS style parallel algorithm.

INPUT
--------------------------------------------------------------------------------

Supported input format:
TXT .lg .txt .ctxt file
ADJ .sadj file
MTX .mtx file
Galois .gr file

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (cmake -DENABLE_HETERO_GALOIS=1 -DENABLE_DIST_GALOIS=1 ../).

2. Run `cd <BUILD>; make -j 

RUN
--------------------------------------------------------------------------------

To run on 1 machine, use the following:
./kclique txt /net/ohm/export/iss/inputs/Mining/mico.lg -k=3 -num_nodes=1 -pset=g -runs=1 -t=12

Output should be like this:

D-Galois Benchmark Suite v5.0.0 (unknown)
Copyright (C) 2018 The University of Texas at Austin
http://iss.ices.utexas.edu/galois/

application: Kcl
Counts the K-Cliques in a graph using BFS extension

Orientation enabled, using DAG
Reading .lg file: /net/ohm/export/iss/inputs/Mining/mico.lg
Number of unique vertex label values: 29
Sorting the neighbor lists... Done
Removing self loops... 0 selfloops are removed
Removing redundent edges... 0 redundent edges are removed
Constructing DAG... 1080156 dag edges are removed
num_vertices 100000 num_edges 1080156
Found 6 devices, using device 0 (GeForce GTX 1080), compute capability 6.1, cores 20*2560.
Launching CUDA TC solver (4220 CTAs, 256 threads/CTA) ...

	total_num_cliques = 12534960

STAT_TYPE, HOST_ID, REGION, CATEGORY, TOTAL_TYPE, TOTAL
STAT, 0, (NULL), Compute, HMAX, 11
STAT, 0, (NULL), GraphReadingTime, HMAX, 7058
PARAM, 0, DistBench, CommandLine, HOST_0, ./kclique txt /net/ohm/export/iss/inputs/Mining/mico.lg -k=3 -num_nodes=1 -pset=g -runs=1 -t=12
PARAM, 0, DistBench, Threads, HOST_0, 12
PARAM, 0, DistBench, Hosts, HOST_0, 1
PARAM, 0, DistBench, Runs, HOST_0, 1
PARAM, 0, DistBench, Run_UUID, HOST_0, 1e19f125-3636-4c90-a02d-fb0b283ee4c5
PARAM, 0, DistBench, Input, HOST_0, /net/ohm/export/iss/inputs/Mining/mico.lg
PARAM, 0, DistBench, PartitionScheme, HOST_0, oec
PARAM, 0, DistBench, Hostname, HOST_0, tuxedo.ices.utexas.edu

