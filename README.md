Overview
========

[![CircleCI](https://circleci.com/gh/IntelligentSoftwareSystems/Galois.svg?style=svg)](https://circleci.com/gh/IntelligentSoftwareSystems/Galois)
[![Build Status](https://travis-ci.org/IntelligentSoftwareSystems/Galois.svg?branch=master)](https://travis-ci.org/IntelligentSoftwareSystems/Galois)

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

Galois is released under the BSD-3-Clause license.


Building Galois
===============

You can checkout the latest release by typing (in a terminal):

```Shell
git clone -b release-5.0 https://github.com/IntelligentSoftwareSystems/Galois
```

The master branch will be regularly updated, so you may try out the latest
development code as well by checking out master branch:

```Shell
git clone https://github.com/IntelligentSoftwareSystems/Galois
```

Dependencies
------------

Galois builds, runs, and has been tested on GNU/Linux. Even though
Galois may build on systems similar to Linux, we have not tested correctness or performance, so please
beware.

At the minimum, Galois depends on the following software:

- A modern C++ compiler compliant with the C++-17 standard (gcc >= 7, Intel >= 19.0.1, clang >= 7.0)
- CMake (>= 3.13)
- Boost library (>= 1.58.0, we recommend building/installing the full library)
- libllvm (>= 7.0 with RTTI support)
- libfmt (>= 4.0)

Here are the dependencies for the optional features:

- Linux HUGE_PAGES support (please see [www.kernel.org/doc/Documentation/vm/hugetlbpage.txt](https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt)). Performance will most likely degrade without HUGE_PAGES
  enabled. Galois uses 2MB huge page size and relies on the kernel configuration to set aside a large amount of 2MB pages. For example, our performance testing machine (4x14 cores, 192GB RAM) is configured to support up to 65536 2MB pages:
  ```Shell
  cat /proc/meminfo | fgrep Huge
  AnonHugePages:    104448 kB
  HugePages_Total:   65536
  HugePages_Free:    65536
  HugePages_Rsvd:        0
  HugePages_Surp:        0
  Hugepagesize:       2048 kB
  ```

- libnuma support. Performance may degrade without it. Please install
  libnuma-dev on Debian like systems, and numactl-dev on Red Hat like systems.
- Doxygen (>= 1.8.5) for compiling documentation as webpages or latex files
- PAPI (>= 5.2.0.0 ) for profiling sections of code
- Vtune (>= 2017 ) for profiling sections of code
- MPICH2 (>= 3.2) if you are interested in building and running distributed system
  applications in Galois
- CUDA (>= 8.0 and < 11.0) if you want to build GPU or distributed heterogeneous applications.
  Note that versions >= 11.0 use an incompatible CUB module and will fail to execute.
- Eigen (3.3.1 works for us) for some matrix-completion app variants


Compiling and Testing Galois
----------------------------
We use CMake to streamline building, testing and installing Galois. In the
following, we will highlight some common commands.

Let's assume that `SRC_DIR` is the directory where the source code for Galois
resides, and you wish to build Galois in some `BUILD_DIR`. Run the following
commands to set up a build directory:

```Shell
SRC_DIR=`pwd` # Or top-level Galois source dir
BUILD_DIR=<path-to-your-build-dir>

mkdir -p $BUILD_DIR
cmake -S $SRC_DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release
```

You can also set up a `Debug` build by running the following instead of the last command above:

```Shell
cmake -S $SRC_DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Debug
```

Galois applications are in `lonestar` directory.  In order to build a particular application:

```Shell
make -C $BUILD_DIR/lonestar/<app-dir-name> -j
# or alternatively
make -C $BUILD_DIR <app-executable-name> -j
# or
cmake --build $BUILD_DIR <app-executable-name> --parallel
```

You can also build everything by running `make -j` in the top-level of build directory, but that may
take a lot of time.

Setting the `BUILD_SHARED_LIBS` to `ON` when calling CMake will make the core runtime library be built as a shared object instead of a static library.

The tests for the core runtime will be built by default when you run `make`
with no target specified. They can be also built explicitly with:

```Shell
make -C $BUILD_DIR/test
```

We provide a few sample inputs that can be downloaded by running:

```Shell
make -C $BUILD_DIR input
```

`make input` will download a tarball of inputs and extract it to
`$BUILD_DIR/inputs/small_inputs` directory. The tarball is downloaded to
`$BUILD_DIR/inputs`

Most of the Galois apps have corresponding tests.
These tests depend on downloading the reference inputs and building the corresponding apps and test binaries.
Once the reference inputs have been downloaded and everything has been built,
the tests for the core library and all the apps can be run by running:

```Shell
make test
# or alternatively
ctest
```

in the build directory.

Capturing Stack Information
---------------------------
Currently if you add `-DSTACK_CAPTURE` to your `cmake` line then you will configure stack capturing.
Please view `libgalois/include/runtime/StackTracer.h` for documentation on functions for printing and reseting.
Do not attempt to modify the capture process otherwise.


Running Galois Applications
===========================

Graph Format
------------

Many Galois/Lonestar applications work with graphs. We store graphs in a binary format
called *galois graph file*
(`.gr` file extension). Other formats such as edge-list or Matrix-Market can be
converted to `.gr` format with `graph-convert` tool provided in galois.
You can build graph-convert as follows:

```Shell
cd $BUILD_DIR
make graph-convert
./tools/graph-convert/graph-convert --help
```

Other applications, such as Delaunay Mesh Refinement may read special file formats
or some may even generate random inputs on the fly.

Running
-------

All Lonestar applications take a `-t` command-line option to specify the number of
threads to use. All applications run a basic sanity check (often insufficient for
correctness) on the program output, which can be turned off with the `-noverify` option. You
can specify `-help` command-line option to print all available options.

Upon successful completion, each application will produce some stats regarding running
time of various sections, parallel loop iterations and memory usage, etc. These
stats are in CSV format and can be redirected to a file using `-statFile` option.
Please refer to the manual for details on stats.

Running LonestarGPU applications
--------------------------

Please refer to `lonestar/analytics/gpu/README.md` and `lonestar/scientific/gpu/README.md` for more details on
compiling and running LonestarGPU applications.

Running Distributed Galois
--------------------------

Please refer to `lonestar/analytics/distributed/README.md` for more details on
running distributed benchmarks.

Documentation
=============

Galois documentation is produced using doxygen, included in this repository, which includes a tutorial, a user's
manual and API documentation for the Galois library.

Users can build doxygen documentation in the build directory using:

```Shell
cd $BUILD_DIR
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
- `lonestardist` contains the source code for the distributed-memory and heterogeneous
  benchmark applications. Please refer to `lonestardist/README.md` for instructions on
  building and running these apps.
- `tools` contains various helper programs such as graph-converter to convert
  between graph file formats and graph-stats to print graph properties

Using Galois as a library
=========================

There are two common ways to use Galois as a library. One way is to copy this
repository into your own CMake project, typically using a git submodule. Then
you can put the following in your CMakeLists.txt:

```CMake
add_subdirectory(galois EXCLUDE_FROM_ALL)
add_executable(app ...)
target_link_libraries(app Galois::shmem)
```

The other common method is to install Galois outside your project and import it
as a package.

If you want to install Galois, assuming that you wish to install it under
`INSTALL_DIR`:

```Shell
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR $SRC_DIR
make install
```

Then, you can put something like the following in CMakeLists.txt:

```CMake
list(APPEND CMAKE_PREFIX_PATH ${INSTALL_DIR})
find_package(Galois REQUIRED)
add_executable(app ...)
target_link_libraries(app Galois::shmem)
```

If you are not using CMake, the corresponding basic commands (although the
specific commands vary by system) are:

```Shell
c++ -std=c++14 app.cpp -I$INSTALL_DIR/include -L$INSTALL_DIR/lib -lgalois_shmem
```
Third-Party Libraries and Licensing
====================

Galois includes some third party libraries that do not use the same license as
Galois. This includes the bliss library (located in lonestar/include/Mining/bliss)
and Modern GPU (located in libgpu/moderngpu). Please be aware of this when
using Galois.

Contact Us
==========
For bugs, please raise an
[issue](https://github.com/IntelligentSoftwareSystems/Galois/issues) on
GiHub.
Questions and comments are also welcome at the Galois users mailing list:
[galois-users@utlists.utexas.edu](galois-users@utlists.utexas.edu). You may
[subscribe here](https://utlists.utexas.edu/sympa/subscribe/galois-users).

If you find a bug, it would help us if you sent (1) the command line and
program inputs and outputs and (2) a core dump, preferably from an executable
built with the debug build.

You can enable core dumps by setting `ulimit -c unlimited` before running your
program. The location where the core dumps will be stored can be determined with
`cat /proc/sys/kernel/core_pattern`.

To create a debug build, assuming you will build Galois in `BUILD_DIR` and the
source is in `SRC_DIR`:

```Shell
cmake -S $SRC_DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Debug
make -C $BUILD_DIR
```

A simple way to capture relevant debugging details is to use the `script`
command, which will record your terminal input and output. For example,

```Shell
script debug-log.txt
ulimit -c unlimited
cat /proc/sys/kernel/core_pattern
make -C $BUILD_DIR <my-app> VERBOSE=1
my-app with-failing-input
exit
```

This will generate a file `debug-log.txt`, which you can send to the mailing
list:[galois-users@utlists.utexas.edu](galois-users@utlists.utexas.edu) for
further debugging or supply when opening a GitHub issue.
