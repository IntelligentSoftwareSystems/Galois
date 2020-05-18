#!/bin/bash

BUILD_DIR=${BUILD_DIR:-/tmp/msan/boost-build}
SOURCE_DIR=${SOURCE_DIR:-/tmp/msan/boost}
BOOST_LIBRARIES=${BOOST_LIBRARIES:-headers,iostreams,serialization}
AS_ROOT=${AS_ROOT:-gosu root}
LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/usr/lib/llvm-10-msan}

set -e

MSAN_LINKER_FLAGS="-lc++abi \
  -Wl,--rpath=${LLVM_INSTALL_PREFIX}/lib \
  -L${LLVM_INSTALL_PREFIX}/lib"
MSAN_FLAGS="-nostdinc++ -stdlib=libc++ \
  -isystem ${LLVM_INSTALL_PREFIX}/include \
  -isystem ${LLVM_INSTALL_PREFIX}/include/c++/v1 \
  ${MSAN_LINKER_FLAGS} \
  -fsanitize=memory\
  -w"

mkdir -p "${SOURCE_DIR}"
curl -fL https://dl.bintray.com/boostorg/release/1.73.0/source/boost_1_73_0.tar.bz2 | tar -xjv -f - -C "${SOURCE_DIR}"
cd "${SOURCE_DIR}/boost_1_73_0"
./bootstrap.sh --with-toolset=clang --with-libraries="${BOOST_LIBRARIES}"
./b2 threading=multi cxxflags="${MSAN_FLAGS}" linkflags="${MSAN_LINKER_FLAGS}" 
${AS_ROOT} ./b2 install
