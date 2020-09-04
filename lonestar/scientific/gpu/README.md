Overview of LonestarGPU Scientific Benchmark Suite
================================================================================

The LonestarGPU suite contains CUDA implementations of several
irregular algorithms that exhibit amorphous data parallelism. Currently,
the LonestarGPU suite contains the following scientific applications,
which can be executed on a single-GPU.

### Scientific Applications
  * Barnes-Hut N-body Simulation
  * Delaunay Mesh Refinement

Compiling LonestarGPU Through CMake 
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
These modules will be available in the ${GALOIS\_ROOT}/external directory

To build the LonestarGPU suite, first, create a build directory and
run CMake with -DGALOIS\_CUDA\_CAPABILITY=\<insert CUDA capability here\>
flag in the build directory. The CUDA capability should be one that your
GPU supports. For example, if you wanted to build for a GTX 1080 and a K80,
the commands would look like this:

```Shell
cd ${GALOIS_ROOT}
mkdir build
cd build
cmake ${GALOIS_ROOT} -DGALOIS_CUDA_CAPABILITY="3.7;6.1"

After compiling through CMake, the system will create the 'lonestar/analytics/gpu'
and 'lonestar/scientific/gpu' directories in ${GALOIS\_ROOT}/build directory. 

Compiling Scientific Applications
================================================================================

Once CMake is completed,  compile the provided scientific apps by executing the 
following command in the ${GALOIS\_ROOT}/build/lonestar/scientific/gpu directory.

`make -j`

You can compile a specific app by executing the following commands
(shown for barneshut).

```Shell
`cd barneshut`
`make -j`
```

Running Scientific Applications
================================================================================

To run a specific app, follow the instructions given in the README.md
in the particular app directory. 

Documentation
================================================================================

Further documentation is available at
[http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu](http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu)




