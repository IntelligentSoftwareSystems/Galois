app_dist(bfs_push)
add_test_dist(bfs_push rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr)

app_dist(bfs_pull)
add_test_dist(bfs_pull rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr)
