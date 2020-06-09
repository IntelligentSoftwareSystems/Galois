app_dist(bfs_push bfs-push)
add_test_dist(bfs-push-dist rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr)

app_dist(bfs_pull bfs-pull)
add_test_dist(bfs-pull-dist rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr)
