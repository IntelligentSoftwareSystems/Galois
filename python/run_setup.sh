#!/bin/bash

ROOT_DIR=$(cd $(dirname $0)/..; pwd)

if [[ "${ROOT_DIR}" != "/source" ]]; then
    echo "This script should be run from within a quick-dev container" >&2
    exit 1
fi

if ! which pkg-config > /dev/null; then
  gosu root apt-get update
  gosu root apt-get install pkg-config
fi

pip3 install cython

mkdir -p "${ROOT_DIR}/build"
if [[ ! -f "${ROOT_DIR}/build/CMakeCache.txt" ]]; then 
  conan install -if "${ROOT_DIR}/build" --build=missing "${ROOT_DIR}/config"
  cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_TOOLCHAIN_FILE="${ROOT_DIR}/build/conan_paths.cmake" \
    -DBUILD_SHARED_LIBS=1
fi
make -C "${ROOT_DIR}/build" -j 4

# Lazy install...
if [[ ! -f "${ROOT_DIR}/build/boost.pc" ]]; then
  conan install -g pkg_config -if "${ROOT_DIR}/build" "${ROOT_DIR}/config"
fi
ln -rfs "${ROOT_DIR}/build/libgalois" "${ROOT_DIR}/python/lib"
ln -rfs "${ROOT_DIR}/libgalois/include" "${ROOT_DIR}/python/include"
export PKG_CONFIG_PATH="${ROOT_DIR}/build"
export LD_LIBRARY_PATH="${ROOT_DIR}/python/lib"

(cd "${ROOT_DIR}/python" && python setup.py build_ext --inplace && python) 

