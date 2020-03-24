app(pagerank-pull PageRank-pull.cpp)
app(pagerank-push PageRank-push.cpp)

add_test_scale(small pagerank-pull -tolerance=0.01 "${BASEINPUT}/scalefree/transpose/rmat10.tgr")
add_test_scale(small-topo pagerank-pull -tolerance=0.01 -algo=Topo "${BASEINPUT}/scalefree/transpose/rmat10.tgr")
add_test_scale(small pagerank-push -tolerance=0.01 "${BASEINPUT}/scalefree/transpose/rmat10.tgr")
add_test_scale(small-sync pagerank-push -tolerance=0.01 -algo=Sync "${BASEINPUT}/scalefree/transpose/rmat10.tgr")
