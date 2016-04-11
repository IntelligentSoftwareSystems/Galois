add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
cuda_add_library(het_auto_gen_cuda gen_cuda.cu)
target_link_libraries(het_auto_gen_cuda ggrt cudadevrt)

app(het_auto_gen gen.cpp DISTSAFE)
target_compile_definitions(het_auto_gen PRIVATE __GALOIS_HET_CUDA__=1)
target_link_libraries(het_auto_gen het_auto_gen_cuda cudadevrt)

