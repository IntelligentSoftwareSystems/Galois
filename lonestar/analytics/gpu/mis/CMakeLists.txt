app_gpu(mis)
if(ENABLE_HETERO_GALOIS)
  target_link_libraries(mis_gpu Galois::gpu -lcurand)
endif()
