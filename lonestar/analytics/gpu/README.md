# LonestarGPU Benchmark Suite

The LonestarGPU suite contains CUDA implementations of several
irregular algorithms that exhibit amorphous data parallelism. Currently, LonestarGPU suite contains the following analytics and scientific applications, which can be executed on a single-GPU.

### Analytics Applications
* Breadth-First Search 
* Connected Components
* Maximal Independent Set
* Minimum Spanning Tree
* Pagerank
* Points-to Analysis
* Single-Source Shortest Paths
* Stochastic Gradient Descent
* Triangle Counting

### Scientific Applications
* Barnes-Hut N-body Simulation
* Delaunay Mesh Refinement

The analytics applications are located in the ${GALOIS\_ROOT}/lonestar/analytics/gpu directory. Whereas the scientific applications are located in the ${GALOIS\_ROOT}/lonestar/scientific/gpu directory

# Compiling LonestarGPU Through CMake 

The dependencies for LonestarGPU suite are the same as shared-memory. Note that  LonestarGPU requires CUDA 8.0 and above. 

To build the LonestarGPU suite, first, create a build directory and run CMake with DGALOIS\_ENABLE\_GPU flag in the build directory, as shown below.

`cd ${GALOIS_ROOT}`
`mkdir build`
`cd build`
`cmake ${GALOIS_ROOT} -DGALOIS_ENABLE_GPU=1`

After compiling through CMake, the system will create the 'lonestar/analytics/gpu' and 'lonestar/scientific/gpu' 
directories in ${GALOIS\_ROOT}/build directory. 

# Compiling Analytics Applications

Once CMake is completed,  compile the provided analytics apps by executing the 
following command in the ${GALOIS\_ROOT}/build/lonestar/analytics/gpu directory.

`make -j`

You can compile a specific app by executing the following commands (shown for bfs).

`cd bfs`
`make -j`

# Running Analytics Applications

To run a specific app, follow the instructions given in the README.md in the particular app directory. 

# Documentation

Further documentation is available at
[http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu](http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu)




