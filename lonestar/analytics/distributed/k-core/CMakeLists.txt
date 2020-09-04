app_dist(kcore_push k-core-push)
add_test_dist(k-core-push-dist rmat15 ${BASEINPUT}/scalefree/symmetric/rmat15.sgr -symmetricGraph -kcore=100)

app_dist(kcore_pull k-core-pull)
add_test_dist(k-core-pull-dist rmat15 ${BASEINPUT}/scalefree/symmetric/rmat15.sgr -symmetricGraph -kcore=100)
