# Contributing to Katana

This repository is currently maintained as a fork the [Galois
project](https://github.com/IntelligentSoftwareSystems/Galois). At some point
in the future, these repositories may diverge. Katana Graph members should
focus their development here.

The instructions in README.md give unopinionated instructions on how to install
and use this repository, which is appropriate to general users. This file
contains instructions and details more appropriate to people who need to
develop and improve the Katana system itself.

The quickest way to start hacking on Katana is to look at
`scripts/setup_dev_ubuntu.sh` and use that as the basis for installing a
development environment on your own machine.

With those developer dependencies installed, you can run the following commands
from the project root directory to build the system:

```shell
case $(uname) in
  Darwin*)
    ;;
  *) conan profile update settings.compiler.libcxx=libstdc++11 default
    ;;
esac
conan remote add kmaragon https://api.bintray.com/conan/kmaragon/conan

mkdir build
cd build
conan install ../config --build=missing
cmake -DCMAKE_TOOLCHAIN_FILE=conan_paths.cmake -DCMAKE_CXX_COMPILER=gcc-9 ..
make
```

Note: Katana builds with a variety of compilers as long as they support the
C++17 standard. See README.md for specific tested versions.

## Specifying and Resolving C++ Dependencies

The above instructions should work if you have installed the C++ library
dependencies in `scripts/setup_dev_ubuntu.sh` (e.g., llvm-dev, arrow) in their
standard system locations (typically `/usr/lib` or `/usr/local/lib`). If you
need to tell `cmake` about additional library locations, you can use the
CMake option `CMAKE_PREFIX_PATH`, as in:

```shell
cmake -DCMAKE_TOOLCHAIN_FILE=conan_paths.cmake \
  -DCMAKE_CXX_COMPILER=gcc-9 \
  -DCMAKE_PREFIX_PATH=<path/to/cmakefiles/for/library>;<another/path> ..
```

As a sidenote, CMake toolchain file is simply a method for initially defining
`CMAKE_PREFIX_PATH` and other CMake options. You can verify this by looking at
the contents of `conan_paths.cmake`.

A common issue is that you have multiple versions of the same dependency,
located in different directories, and CMake picks the wrong version.

The process by which CMake finds packages is involved, and the [CMake
documentation](https://cmake.org/cmake/help/latest/command/find_package.html#search-procedure)
contains all the gory details. One implication, though, is that CMake adds
directories in your path to its set of search locations.

Thus, if the LLVM C++ compiler (clang++) is in your path, CMake will attempt to
use the LLVM support libraries (e.g., libLLVMSupport.a, libclang.so) associated
with your compiler installation by default, even though your compiler and the
version of the LLVM support libraries you use are not strictly related to each
other.

You can work around this by putting the location of the LLVM support libraries
in `CMAKE_PREFIX_PATH` because that takes precedence over locations in your
path. Alternatively, you can indicate the location of the LLVM libraries
directly with `LLVM_DIR`:

```shell
cmake -DCMAKE_TOOLCHAIN_FILE=conan_paths.cmake \
  -DLLVM_DIR="$(llvm-config-X --cmakedir)" ..
```

# Submitting Changes

The way to contribute code to the Katana project is to make a fork of this
repository and create a pull request with your desired changes. For a general
overview of this process, this
[description](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)
from Atlassian.

# Some Helpful Commands

```shell
make -C <some subdirectory in the build tree>
```

This builds only the targets defined in that subdirectory (and their
dependencies). This is useful for compiling a single application or library at
a time.

```shell
make -j 4
```

This launches `4` jobs to build the system concurrently.

```shell
make VERBOSE=1
```

This prints out the commands that are run. This is useful for debugging
installation issues.

```shell
ctest -L quick
```

This runs code tests locally on your machine.  Most of the tests require sample
graphs. You can get them with `make inputs`.

# Some Helpful tools

[ccmake](https://cmake.org/cmake/help/v3.0/manual/ccmake.1.html): a simple
terminal UI for configuring cmake projects. This is helpful for setting
various build options without having to remember their names exactly.

[ccache](https://ccache.dev/): a wrapper around compilation commands to reuse
object files even if their timestamps change as long as the contents of the
source file and compiler flags are the same. To enable use: `cmake
-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`

The LLVM project provides many useful development tools for C++:
[clang-format](https://clang.llvm.org/docs/ClangFormat.html),
[clang-tidy](https://clang.llvm.org/extra/clang-tidy/), and
[clangd](https://clangd.llvm.org/).  Most IDEs can be configured to use these
tools automatically.
