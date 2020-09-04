app_dist(matrixCompletion matrixcompletion NO_GPU)
add_test_dist(matrixcompletion-dist Epinions_dataset NO_ASYNC NO_GPU ${BASEINPUT}/weighted/bipartite/Epinions_dataset.gr -graphTranspose=${BASEINPUT}/weighted/bipartite/Epinions_dataset.tgr -maxIterations=2)
