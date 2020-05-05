#!/bin/bash

BUILD_DIR=${BUILD_DIR:-/source/build}
SOURCE_DIR=${SOURCE_DIR:-/source}
LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/usr/lib/llvm-10-msan}

MSAN_LINKER_FLAGS="-lc++abi \
  -Wl,--rpath=${LLVM_INSTALL_PREFIX}/lib \
  -L${LLVM_INSTALL_PREFIX}/lib"

MSAN_FLAGS="-nostdinc++ -stdlib=libc++ \
  -isystem ${LLVM_INSTALL_PREFIX}/include \
  -isystem ${LLVM_INSTALL_PREFIX}/include/c++/v1 \
  ${MSAN_LINKER_FLAGS} \
  -fsanitize=memory \
  -w"

cmake \
  -DCMAKE_PREFIX_PATH="${LLVM_INSTALL_PREFIX}" \
  -DGALOIS_USE_SANITIZER=MemoryWithOrigins \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_FLAGS="${MSAN_FLAGS}" \
  -DCMAKE_C_FLAGS="${MSAN_FLAGS}" \
  -DCMAKE_EXE_LINKER_FLAGS="${MSAN_LINKER_FLAGS}" \
  -S "${SOURCE_DIR}" \
  -B "${BUILD_DIR}"
