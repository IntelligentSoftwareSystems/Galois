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

For heterogeneous (which is also distributed) Galois, i.e. D-IrGL:

`cmake ${GALOIS_ROOT} -DGALOIS_ENABLE_DIST=1 -DGALOIS_ENABLE_GPU=1`

Note that heterogeneous Galois requires CUDA 8.0 and above and a compiler
that is compatible with the CUDA version that you use, and also note that
enabling heterogeneous Galois also activates distributed Galois.

Compiling with distributed/heterogeneous Galois will add the 'dist_apps' 
directory to the build folder. Compiling with heterogeneous Galois enabled with 
enable certain options in 'dist_apps'.

Compiling Provided Apps
================================================================================

Once CMake is successfully completed, you can build the provided apps with the 
following command in the dist_apps directory.

`make -j`

You can compile a specific app by specifying it by name:

`make -j bfs_push`

Running Provided Apps
================================================================================

You can learn how to run compiled applications by running them with the -help
command line option:

`./bfs_push -help`

Most of the provided graph applications take graphs in a .gr format, which
is a Galois graph format that stores the graph in a CSR or CSC format. We 
provide a graph converter tool under 'tools/graph-convert' that can take
various graph formats and convert them to the Galois format.

Running Provided Apps (Distributed Apps)
================================================================================

First, note that if running multiple processes on a single machine, specifying
`GALOIS_DO_NOT_BIND_THREADS=1` as an environment variable is crucial for 
performance.

If using MPI, multiple processes split across multiple hosts can be specified
with the following:

`GALOIS_DO_NOT_BIND_THREADS=1 mpirun -n=<# of processes> -hosts=<machines to run on> ./bfs_push <input graph>`

The distributed applications have a few common command line flags that are
worth noting. More details can be found by running a distributed application
with the -help flag.

`-partition=<partitioning policy>`

Specifies the partitioning that you would like to use when splitting the graph
among multiple hosts.

`-graphTranspose`

Specifies the transpose of the provided input graph. This is used to 
create certain partitions of the graph (and is required for some of the 
partitioning policies). It also makes 

`-runs`

Number of times to run an application.

`-statFile`

Specify the file in which to output run statistics to.

`-t`

Number of threads to use on a single machine excluding a communication thread
that is used by all of the provided distributed benchmarks. Note that 
GPUs only use 1 thread (excluding the communication thread).

`-verify`

Outputs a file with the result of running the application. For example, 
specifying this flag on a bfs application will output the shortest distances
to each node.

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
"g" (GPUs). For example, if I have 2 machines with 8 GPUs each, 
but I want to run with 3 GPUs on each machine, I would use `-pset="ggg"`. 
Therefore, combined with `-num_nodes=2`, I would have a total of 6 units of 
execution: 3 GPUs on 2 machines for a total of 6. This creates a total of
6 processes across the 2 machines (1 for each GPU).

Also, it suffices to use only one "c" in pset to run on CPUs on your machines: 
you can specify the amount of CPUs to use using the aforementioned thread 
option `-t`.

Basic Use (Creating Your Own Applications)
================================================================================

You can run the sample applications and make your own Galois programs directly
in the build tree without installing anything. Just add a subdirectory under
dist_apps, copy a CMakeLists.txt file from another application to your new
application, and add the subdirectory to the CMakeLists in dist_apps.
