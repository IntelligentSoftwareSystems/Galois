#!/bin/bash
#
# Build and install libc++abi, libc++ and ${LLVM_COMPONENTS} with memory
# sanitization.
set -e

NUM_PARALLEL=${NUM_PARALLEL:-2}
BUILD_DIR=${BUILD_DIR:-/tmp/msan/llvm-build}
SOURCE_DIR=${SOURCE_DIR:-/tmp/msan/llvm}
LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/usr/lib/llvm-10-msan}
LLVM_COMMIT=${LLVM_COMMIT:-release/10.x}
LLVM_COMPONENTS=${LLVM_COMPONENTS-LLVMSupport}
AS_ROOT=${AS_ROOT:-gosu root}

git clone -b ${LLVM_COMMIT} --depth 1 https://github.com/llvm/llvm-project.git "${SOURCE_DIR}"

mkdir -p "${BUILD_DIR}/libcxxabi"
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX}" \
  -DLIBCXXABI_LIBCXX_INCLUDES="${SOURCE_DIR}/libcxx/include" \
  -DLLVM_PATH="${SOURCE_DIR}" \
  -S "${SOURCE_DIR}/libcxxabi" \
  -B "${BUILD_DIR}/libcxxabi"
cmake --build "${BUILD_DIR}/libcxxabi" --parallel "${NUM_PARALLEL}"
${AS_ROOT} cmake --build "${BUILD_DIR}/libcxxabi" --target install

# Bootstrap llvm build with memory sanitized libcxx.

MSAN_LINKER_FLAGS="-lc++abi \
  -Wl,--rpath=${LLVM_INSTALL_PREFIX}/lib \
  -L${LLVM_INSTALL_PREFIX}/lib"

mkdir -p "${BUILD_DIR}/libcxx"
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_SHARED_LINKER_FLAGS="${MSAN_LINKER_FLAGS}" \
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX}" \
  -DLIBCXX_CXX_ABI_INCLUDE_PATHS=${SOURCE_DIR}/libcxxabi/include \
  -DLIBCXX_CXX_ABI=libcxxabi \
  -DLLVM_PATH="${SOURCE_DIR}" \
  -DLLVM_USE_SANITIZER=MemoryWithOrigins \
  -S "${SOURCE_DIR}/libcxx" \
  -B "${BUILD_DIR}/libcxx"
cmake --build "${BUILD_DIR}/libcxx" --parallel "${NUM_PARALLEL}"
${AS_ROOT} cmake --build "${BUILD_DIR}/libcxx" --target install

# Build llvm libraries
#
# -fsanitize and -stdlib=c++ are required here in addition to CMake below
# because even linking test programs with libc++-msan requires
# -fsanitize=memory.
MSAN_FLAGS="-nostdinc++ -stdlib=libc++ \
  -isystem ${LLVM_INSTALL_PREFIX}/include \
  -isystem ${LLVM_INSTALL_PREFIX}/include/c++/v1 \
  ${MSAN_LINKER_FLAGS} \
  -fsanitize=memory \
  -w"
mkdir -p "${BUILD_DIR}/llvm"
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_FLAGS="${MSAN_FLAGS}" \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="${MSAN_FLAGS}" \
  -DCMAKE_EXE_LINKER_FLAGS="${MSAN_LINKER_FLAGS}" \
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX}" \
  -DLLVM_ENABLE_LIBCXX=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_USE_SANITIZER=MemoryWithOrigins \
  -S "${SOURCE_DIR}/llvm" \
  -B "${BUILD_DIR}/llvm"
cmake --build "${BUILD_DIR}/llvm" --parallel "${NUM_PARALLEL}" --target ${LLVM_COMPONENTS}
for c in ${LLVM_COMPONENTS}; do
  ${AS_ROOT} cmake -DCOMPONENT=${c} -P "${BUILD_DIR}/llvm/cmake_install.cmake"
done
