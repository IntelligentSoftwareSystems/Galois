app_analy_gpu(mis maximal-independentset)
set_property(TARGET maximal-independentset-gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(maximal-independentset-gpu ${CUDA_cudadevrt_LIBRARY})
target_link_libraries(maximal-independentset-gpu Galois::gpu -lcurand)
