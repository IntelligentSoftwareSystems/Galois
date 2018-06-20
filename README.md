Getting Started {#gettingStarted}
====================

Galois is a C++ library designed to ease parallel programming, especially for
applications with irregular parallelism (e.g., irregular amount of work in parallel
sections, irregular memory accesses and branching patterns). It implements
an implicitly parallel programming model, where the programmer replaces serial loop
constructs (e.g. for and while) and serial data structures in their algorithms with parallel loop
constructs and concurrent data structures provided by Galois to express their algorithms.
Galois is designed so that the programmer does not have to deal with low-level parallel programming constructs such as
threads, locks, barriers, condition variables, etc. 

Highlights include:
- Parallel *for_each* loop that handles dependencies between iterations, as well as
  dynamic work creation, and a *do_all* loop for simple parallelism. Both provide load balancing and excellent
  scalability on multi-socket systems
- A concurrent graph library designed for graph analytics algorithms as well as
  other domains such as irregular meshes. 
- Scalable concurrent containers such as bag, vector, list, etc. 

Galois is released under article BSD-3-Clause license. 


Building Galois
===========================

Dependencies
--------------

Galois builds and runs on GNU/Linux (and compatible operating systems). 

At the minimum, Galois depends on the following software:

- A modern C++ compiler compliant with the C++-14 standard (GCC >= 6.1, Intel >= 17.0, LLVM >= 4.0)
- CMake (>= 3.2.3)
- Boost library ( >= 1.60.0, we recommend building/installing the full library)


Here are the dependencies for the optional features: 

- MPICH2 (>= 3.2) if you are interested in building and running distributed system
  applications in Galois
- CUDA (>= 8.0) if you want to build distributed hetergeneous applications
- Eigen (3.3.1 works for us) for some matrix completion variants
- Doxygen (>= 1.8.5) for compiling documentation as webpages or latex files


Compiling Galois
--------------------------
We use CMake. Run the following commands to set up a build directory, e.g. `build/default`, or `build/debug`, etc.:

```Shell
ROOT=`pwd` # Or top-level Galois source dir
mkdir -p build/default; cd build/default; cmake ${ROOT}
```

or

```Shell
mkdir -p build/debug; cd build/debug; cmake -DCMAKE_BUILD_TYPE=Debug ${ROOT}
```

Galois applications are in `lonestar` directory.  In order to build a particular application:

```Shell
cd lonestar/<app-dir-name>; make -j
```

You can also build everything by running `make -j` in the build directory, but that may
take a lot of time and will download additional files.

More esoteric systems may require a toolchain file; check `../cmake/Toolchain`
if there is a file corresponding to your system. If so, use the following
CMake command:

```Shell
cmake -C ${ROOT}/cmake/Toolchain/${platform}-tryrunresults.cmake \
  -DCMAKE_TOOLCHAIN_FILE=${ROOT}/cmake/Toolchain/${platform}.cmake ${ROOT}
```


Running Galois Applications
=============================

Inputs
-------

Many Galois/Lonestar applications work with graphs. We store graphs in a binary format
called *galois graph file* 
(`.gr` file extension). Other formats such as edge-list or Matrix-Market can be
converted to `.gr` format with `graph-convert` tool provided in galois. 
You can run in your build directory:

```Shell
make graph-convert
./tools/graph-convert --help
```

Other applications, such as Delaunay Mesh Refinement may read special file formats
or some may even generate random inputs on the fly. 

Running
---------

All Lonestar applications take a `-t` command-line option to specify the number of
threads to use. All applications run a basic sanity check (often insufficient for
correctness) on the program output, which can be turned off with the `-noverify` option. 

Upon successful completion, each application will produce some stats regarding running
time of various sections, parallel loop iterations and memory usage, etc. These
stats are in CSV format and can be redirected to a file using `-statFile` option.
Please refer to the manual for details on stats. 

Running Distributed Galois
---------

Please refer to README-DIST in the dist_apps directory for more details on
running distributed benchmarks.

Documentation
====================

Galois documentation is produced using doxygen, included in this repository, which includes a tutorial, a user's
manual and API documentation for the Galois library. 

Users can build doxygen documentation in the build directory using:

```Shell
make doc
your-fav-browser html/index.html &
```

See online documentation at:
 [http://iss.ices.utexas.edu/?p=projects/galois](http://iss.ices.utexas.edu/?p=projects/galois)


Source-Tree Organization
========================

- `libgalois` contains the source code for the shared-memory Galois library, e.g., runtime, graphs, worklists, etc. 
- `lonestar` contains the Lonestar benchmark applications and tutorial examples for Galois
- `libdist` contains the source code for the distributed-memory and heterogeneous Galois library
- `dist_apps` contains the source code for the distributed-memory and heterogeneous
  benchmark applications
- `tools` contains various helper programs such as graph-converter to convert
  between graph file formats and graph-stats to print graph properties



Installing Galois as a library
==============================
If you want to install Galois as a library,

```Shell
cmake -DCMAKE_INSTALL_PREFIX=${installdir} ${ROOT}
make install
```

or, to speed up compilation,

```Shell
cmake -DCMAKE_INSTALL_PREFIX=${installdir} -DSKIP_COMPILE_APPS=1 ${ROOT}
make install
```


Using Installed Galois
-------------------------
If you are using CMake, put something like the following CMakeLists.txt:

```CMake
set(CMAKE_PREFIX_PATH ${installdir}/lib/cmake/Galois ${CMAKE_PREFIX_PATH})
find_package(Galois REQUIRED)
include_directories(${Galois_INCLUDE_DIRS})
set(CMAKE_CXX_COMPILER ${Galois_CXX_COMPILER})
set(CMAKE_CXX_FLAGS  "${Galois_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
add_executable(app ...)
target_link_libraries(app ${Galois_LIBRARIES})
```

Using basic commands (although the specific commands vary by system):

```Shell
c++ -std=c++14 app.cpp -I${installdir}/include -L${installdir}/lib -lgalois_shmem
```
