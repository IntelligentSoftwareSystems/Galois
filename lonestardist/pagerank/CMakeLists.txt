app_dist(pagerank_pull)
add_test_dist(pagerank_pull rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -maxIterations=5 -tolerance=1)

app_dist(pagerank_push)
add_test_dist(pagerank_push rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -maxIterations=5 -tolerance=1)
