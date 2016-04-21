# for debugging
add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)

cuda_add_library(het_auto_gen_cuda gen_cuda.cu)
target_link_libraries(het_auto_gen_cuda ggrt)

app(het_auto_gen gen.cpp DISTSAFE)
target_compile_definitions(het_auto_gen PRIVATE __GALOIS_HET_CUDA__=1)
target_link_libraries(het_auto_gen het_auto_gen_cuda)

