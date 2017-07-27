#!/bin/bash
#Common variables:: 
LLVM_BUILD_DIR=/net/dyne/workspace/rashid/llvm37/build/
GALOIS_DIST_SRC_DIR=/h2/rashid/workspace/GaloisDist/gdist/
GALOIS_DIST_BUILD_DIR=/net/faraday/workspace/rashid/GaloisDist/release/
CUDA_HOME=/org/centers/cdgc/cuda/cuda-7.0/
IN_FILE_NAME=$2
USE_CASE=$1


GALOIS_INCLUDE_DIRS="-I${BOOST_INC} -I${GALOIS_DIST_SRC_DIR}/exp/include -I${MPI_DIR}/include -I${GALOIS_DIST_BUILD_DIR}/include -I${GALOIS_DIST_SRC_DIR}/include -I${GALOIS_DIST_SRC_DIR}/libruntime/include/ -I${GALOIS_DIST_SRC_DIR}/libsubstrate/include/ -I${GALOIS_DIST_SRC_DIR}/lonestar/include/ -I${GALOIS_DIST_SRC_DIR}/libllvm/include -I${GALOIS_DIST_SRC_DIR}/libgraphs/include -I${GALOIS_DIST_SRC_DIR}/libnet/include -I${CUDA_HOME}/include"

#For Analysis: 

#${LLVM_BUILD_DIR}/bin/clang++ -Xclang -load -Xclang ${LLVM_BUILD_DIR}/lib/GaloisFunctionsAnalysis.so -Xclang -plugin -Xclang galois-analysis  -DGALOIS_USE_EXP -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -g -gcc-toolchain /net/faraday/workspace/local/modules/gcc-4.9/bin/.. -fcolor-diagnostics -std=c++11 -O3 -DNDEBUG -I${BOOST_INC} -I${GALOIS_DIST_SRC_DIR}/exp/include -I${MPI_DIR}/include -I${GALOIS_DIST_BUILD_DIR}/include -I${GALOIS_DIST_SRC_DIR}/include -I${GALOIS_DIST_SRC_DIR}/libruntime/include/ -I${GALOIS_DIST_SRC_DIR}/libsubstrate/include/ -I${GALOIS_DIST_SRC_DIR}/lonestar/include/ -I${GALOIS_DIST_SRC_DIR}/libllvm/include -I${GALOIS_DIST_SRC_DIR}/libgraphs/include -I${GALOIS_DIST_SRC_DIR}/libnet/include -I${CUDA_HOME}/include -o CMakeFiles/SGD_gen.dir/pageRankPull_gen.cpp.o -c ${IN_FILE_NAME}


#analysis:

if [ "${USE_CASE}" == "analysis" ] 
then
${LLVM_BUILD_DIR}/bin/clang++ -Xclang -load -Xclang ${LLVM_BUILD_DIR}/lib/GaloisFunctionsAnalysis.so -Xclang -plugin -Xclang galois-analysis  -DGALOIS_USE_EXP -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -g -gcc-toolchain /net/faraday/workspace/local/modules/gcc-4.9/bin/.. -fcolor-diagnostics -std=c++11 -O3 -DNDEBUG ${GALOIS_INCLUDE_DIRS} -o CMakeFiles/SGD_gen.dir/pageRankPull_gen.cpp.o -c ${IN_FILE_NAME}

elif [ "${USE_CASE}" == "codegen" ] 
then
#codegen:
${LLVM_BUILD_DIR}/bin/clang++ -Xclang -load -Xclang ${LLVM_BUILD_DIR}/lib/GaloisFunctions.so -Xclang -plugin -Xclang galois-fns  -DGALOIS_USE_EXP -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -g -gcc-toolchain /net/faraday/workspace/local/modules/gcc-4.9/bin/.. -fcolor-diagnostics -std=c++11 -O3 -DNDEBUG ${GALOIS_INCLUDE_DIRS} -o CMakeFiles/SGD_gen.dir/pageRankPull_gen.cpp.o -c ${IN_FILE_NAME}

elif [ "${USE_CASE}" ==  "opencl" ] 
then
${LLVM_BUILD_DIR}/bin/clang++ -Xclang -load -Xclang ${LLVM_BUILD_DIR}/lib/OpenCLCodeGenHost.so -Xclang -plugin -Xclang opencl-analysis  -DGALOIS_USE_EXP -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -g -gcc-toolchain /net/faraday/workspace/local/modules/gcc-4.9/bin/.. -fcolor-diagnostics -std=c++11 -O3 -DNDEBUG ${GALOIS_INCLUDE_DIRS} -o CMakeFiles/SGD_gen.dir/pageRankPull_gen.cpp.o -c ${IN_FILE_NAME}

elif [ "${USE_CASE}" ==  "clcodegen" ] 
then
${LLVM_BUILD_DIR}/bin/clang++ -Xclang -load -Xclang ${LLVM_BUILD_DIR}/lib/OpenCLCodeGenDevice.so -Xclang -plugin -Xclang opencl-device-codegen  -DGALOIS_USE_EXP -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -g -gcc-toolchain /net/faraday/workspace/local/modules/gcc-4.9/bin/.. -fcolor-diagnostics -std=c++11 -O3 -DNDEBUG ${GALOIS_INCLUDE_DIRS} -o CMakeFiles/SGD_gen.dir/pageRankPull_gen.cpp.o -c ${IN_FILE_NAME}

elif [ "${USE_CASE}" ==  "aos2soa" ] 
then
${LLVM_BUILD_DIR}/bin/clang++ -Xclang -load -Xclang ${LLVM_BUILD_DIR}/lib/AosToSoaPlugin.so -Xclang -plugin -Xclang aos2soa  -DGALOIS_USE_EXP -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -g -gcc-toolchain /net/faraday/workspace/local/modules/gcc-4.9/bin/.. -fcolor-diagnostics -std=c++11 -O3 -DNDEBUG ${GALOIS_INCLUDE_DIRS} -o CMakeFiles/SGD_gen.dir/pageRankPull_gen.cpp.o -c ${IN_FILE_NAME}

elif [ "${USE_CASE}" == "astdump" ]
then

${LLVM_BUILD_DIR}/bin/clang -cc1 -ast-dump -DGALOIS_USE_EXP -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -fcolor-diagnostics -std=c++11 -O3 -DNDEBUG ${GALOIS_INCLUDE_DIRS} -o CMakeFiles/SGD_gen.dir/pageRankPull_gen.cpp.o ${IN_FILE_NAME}
fi
