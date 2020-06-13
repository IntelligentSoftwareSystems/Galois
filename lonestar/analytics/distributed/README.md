Overview of Distributed and Heterogeneous Systems in Galois
================================================================================

This directory contains benchmarks that run using D-Galois and D-IrGL.

D-Galois is distributed Galois built using the Gluon communication substrate.
Similarly, D-IrGL is distributed IrGL built using Gluon.
Gluon is just the communication substrate: it is not a standalone system.

Basic Compiling Through CMake (Distributed and Heterogeneous Galois)
================================================================================

The dependencies for distributed Galois are exactly the same as shared-memory
Galois except that it requires an MPI library (e.g. mpich2) to be on the
system as well.

To build distributed/heterogeneous Galois, certain CMake flags must be
specified.

For distributed Galois, i.e. D-Galois:

`cmake ${GALOIS_ROOT} -DGALOIS_ENABLE_DIST=1`

For distributed and heterogeneous Galois, i.e. D-IrGL:

`cmake ${GALOIS_ROOT} -DGALOIS_ENABLE_DIST=1 -DGALOIS_CUDA_CAPABILITY=<insert CUDA capability here>`

The CUDA capability should be one that your GPU supports. For example, if you
wanted to use a GTX 1080 and a K80, the command would look like this:

`cmake ${GALOIS_ROOT} -DGALOIS_ENABLE_DIST=1 -DGALOIS_CUDA_CAPABILITY="3.7;6.1"`

Note that heterogeneous Galois requires CUDA 8.0 and above and a compiler
that is compatible with the CUDA version that you use.

Note that heterogeneous Galois requires the cub and moderngpu git submodules, which can be cloned using the followed commands.

```Shell
cd $GALOIS_ROOT
git submodule init
git submodule update 
```
These modules will be cloned in the ${GALOIS\_ROOT}/external directory

Compiling with distributed Galois will add the `distributed` directory under
`lonestar/analytics` to the build folder.

Compiling Provided Apps
================================================================================

Once CMake is successfully completed, you can build the provided apps with the
following command in lonestar/analytics/distributed directory.

`make -j`

You can compile specific apps by going their directories and running make.

Running Provided Apps
================================================================================

You can learn how to run compiled applications by running them with the -help
command line option:

`./bfs-push -help`

Most of the provided graph applications take graphs in a .gr format, which
is a Galois graph format that stores the graph in a CSR or CSC format. We
provide a graph converter tool under 'tools/graph-convert' that can take
various graph formats and convert them to the Galois format.

Running Provided Apps (Distributed Apps)
================================================================================

First, note that if running multiple processes on a single machine (e.g.,
single-host multi-GPU or multi-host multi-GPU where a process is spawned for
each GPU), specifying `GALOIS_DO_NOT_BIND_THREADS=1` as an environment variable
is crucial for performance.

If using MPI, multiple processes split across multiple hosts can be specified
with the following:

`GALOIS_DO_NOT_BIND_THREADS=1 mpirun -n=<# of processes> -hosts=<machines to run on> ./bfs-push <input graph>`

The distributed applications have a few common command line flags that are
worth noting. More details can be found by running a distributed application
with the -help flag.

`-partition=<partitioning policy>`

Specifies the partitioning that you would like to use when splitting the graph
among multiple hosts.

`-exec=Sync,Async`

Specifies synchronous communication (bulk-synchronous parallel where every host
blocks for messages from other hosts at the end of a round of execution)
or asynchronous communication (bulk-asynchronous parallel where a host does
not have to block on messages from other hosts at the end of the round and
may continue execution).

`-graphTranspose`

Specifies the transpose of the provided input graph. This is used to
create certain partitions of the graph (and is required for some of the
partitioning policies).

`-runs`

Number of times to run an application.

`-statFile`

Specify the file in which to output run statistics to.

`-t`

Number of threads to use on a single machine excluding a communication thread
that is used by all of the provided distributed benchmarks. Note that
GPUs only use 1 thread (excluding the communication thread).

`-output` / `-outputLocation=<directory>`

Outputs the result of running the application to a file. For example,
specifying this flag on a bfs application will output the shortest distances to
each node.

Running Provided Apps (Distributed Heterogeneous Apps)
================================================================================

Heterogeneous apps have additional command line parameters:

`-num_nodes=<num>`

Specifies the total number of PHYSICAL machines on the system. For example,
you could have 2 machines with 8 GPUs each for a total of 16 processes,
but you still would only have 2 machines. Therefore, you would use
`-num_nodes=2`. Note that there **must** be one process per GPU in use.

`-pset=<string>`

Specifies the architecture to run on on a single machine using "c" (CPUs) and
"g" (GPUs). For example, if you have 2 machines with 8 GPUs each,
but you want to run with 3 GPUs on each machine, you would use `-pset="ggg"`.
Therefore, combined with `-num_nodes=2`, you would have a total of 6 units of
execution: 3 GPUs on 2 machines for a total of 6. This creates a total of
6 processes across the 2 machines (1 for each GPU).

Also, it suffices to use only one "c" in pset to run on CPUs on your machines:
you can specify the amount of cores/hyperthreads to use using the
aforementioned thread option `-t`.

Examples for Running Provided Apps
================================================================================

To run 3 processes all on a single machine, use the following:
`GALOIS_DO_NOT_BIND_THREADS=1 mpirun -n=3 ./bfs_push rmat15.gr -graphTranspose=rmat15.tgr -t=4 -num_nodes=1 -partition=oec`
Note: when heterogeneous execution is not enabled via `GALOIS_CUDA_CAPABILITY`,
`-num_nodes=1` is invalid and will not appear as an option.

is not correct if heterogeneous execution is not
enabled via specifying the CUDA capability (as it does not appear as an option if
heterogeneous execution is not on).

To run on 3 CPUs on h1, h2, and h3, use the following:
`mpirun -n=3 -hosts=h1,h2,h3 ./cc_push rmat15.sgr -symmetricGraph -t=1 -num_nodes=1 -partition=iec`

To run on 3 GPUs on a single machine, use the following:
`GALOIS_DO_NOT_BIND_THREADS=1 mpirun -n=3 ./sssp_pull rmat15.gr -graphTranspose=rmat15.tgr -t=1 -num_nodes=1 -pset="ggg" -partition=cvc`

To run on 4 GPUs on 2 machines h1 and h2 (each with 2 GPUs), use the following:
`GALOIS_DO_NOT_BIND_THREADS=1 mpirun -n=4 -hosts=h1,h2 ./bfs_pull rmat15.gr -graphTranspose=rmat15.tgr -t=1 -num_nodes=2 -pset="gg" -partition=cvc-iec`
Note that `mpirun -n=4` is 4 because there are a total of 4 execution units being used.

To run on 1 CPU and 1 GPU each on 2 machines h1 and h2, use the following:
`GALOIS_DO_NOT_BIND_THREADS=1 mpirun -n=4 -hosts=h1,h2 ./pagerank_pull rmat15.gr -graphTranspose=rmat15.tgr -t=1 -num_nodes=2 -pset="cg" -partition=oec`

Performance Considerations
================================================================================

* As mentioned above if running multiple processes on a single machine,
  specifying `GALOIS_DO_NOT_BIND_THREADS=1` as an environment variable is
  crucial for performance.

* We have also observed that `GALOIS_DO_NOT_BIND_THREADS=1` to improve
  performance in a distributed setting as well (multiple hosts each with its
  own process).

* For 16 or less hosts/GPUs, for performance, we recommend using an
  **edge-cut** partitioning policy (OEC or IEC) with **synchronous**
  communication for performance.

* For 32 or more hosts/GPUs, for performance, we recommend using the
  **Cartesian vertex-cut** partitioning policy (CVC) with **asynchronous**
  communication for performance.

Publications Related to Distributed Applications
================================================================================

Please see the publications listed below for information on the distributed
runtime as well as performance studies we have conducted over the years.

Roshan Dathathri, Gurbinder Gill, Loc Hoang, Hoang-Vu Dang, Alex Brooks,
Nikoli Dryden, Marc Snir, Keshav Pingali, “Gluon: A Communication-Optimizing
Substrate for Distributed Heterogeneous Graph Analytics,” Proceedings of the
39th ACM SIGPLAN Conference on Programming Language Design and Implementation
(PLDI), June 2018.

Gurbinder Gill, Roshan Dathathri, Loc Hoang, Andrew Lenharth, Keshav Pingali,
“Abelian: A Compiler for Graph Analytics on Distributed, Heterogeneous
Platforms,” Proceedings of the 24th International European Conference on
Parallel and Distributed Computing (Euro-Par), August 2018.

Gurbinder Gill, Roshan Dathathri, Loc Hoang, Keshav Pingali, “A Study of
Partitioning Policies for Graph Analytics on Large-scale Distributed
Platforms,” Proceedings of the 45th International Conference on Very Large Data
Bases (PVLDB), 12(4): 321-334, December 2018.

Loc Hoang, Matteo Pontecorvi, Roshan Dathathri, Gurbinder Gill, Bozhi You,
Keshav Pingali, Vijaya Ramachandran, “A Round-Efficient Distributed
Betweenness Centrality Algorithm,” Proceedings of the 24th ACM SIGPLAN
Symposium on Principles and Practice of Parallel Programming (PPoPP), February
2019.

Roshan Dathathri, Gurbinder Gill, Loc Hoang, Keshav Pingali, “Phoenix: A
Substrate for Resilient Distributed Graph Analytics,” Proceedings of the 24th
ACM International Conference on Architectural Support for Programming Languages
and Operating Systems (ASPLOS), April 2019.

Loc Hoang, Roshan Dathathri, Gurbinder Gill, Keshav Pingali, “CuSP: A
Customizable Streaming Edge Partitioner for Distributed Graph Analytics,”
Proceedings of the 33rd IEEE International Parallel and Distributed Processing
Symposium (IPDPS), May 2019.

Loc Hoang, Vishwesh Jatala, Xuhao Chen, Udit Agarwal, Roshan Dathathri,
Gurbinder Gill, Keshav Pingali, “DistTC: High Performance Distributed Triangle
Counting,” Proceedings of the IEEE International Conference on High Performance
Extreme Computing (HPEC), September 2019.

Roshan Dathathri, Gurbinder Gill, Loc Hoang, Hoang-Vu Dang, Vishwesh Jatala, V.
Krishna Nandivada, Marc Snir, Keshav Pingali, “Gluon-Async: A Bulk-Asynchronous
System for Distributed and Heterogeneous Graph Analytics,” Proceedings of the
28th IEEE International Conference on Parallel Architectures and Compilation
Techniques (PACT), September 2019.

Vishwesh Jatala, Roshan Dathathri, Gurbinder Gill, Loc Hoang, V. Krishna
Nandivada, Keshav Pingali, “A Study of Graph Analytics for Massive Datasets on
Distributed GPUs,” Proceedings of the 34th IEEE International Parallel and
Distributed Processing Symposium (IPDPS), May 2020.

Basic Use (Creating Your Own Applications)
================================================================================

You can run the sample applications and make your own Galois programs directly
in the build tree without installing anything. Just add a subdirectory under
distributed, copy a CMakeLists.txt file from another application to your new
application, and add the subdirectory to the CMakeLists in distributed.
