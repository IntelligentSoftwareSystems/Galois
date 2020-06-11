app_scientific_gpu(dmr delaunayrefinement)
#add_test_gpu(delaunayrefinement rmat15 rmat15.out dmr ${BASEINPUT}/reference/meshes/r10k.1 20)
add_test_gpu(delaunayrefinement rmat15 rmat15.out dmr ${BASEINPUT}/meshes/250k.2 20)
