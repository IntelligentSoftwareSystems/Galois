app_analy_gpu(sgd matrixcompletion)
#add_test_gpu(matrixcompletion Epinions Epinions.out sgd ${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr)
add_test_gpu(matrixcompletion bgg bgg.out sgd)
