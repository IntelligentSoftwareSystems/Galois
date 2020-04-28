app_dist(kcore_push)
add_test_dist(kcore_push rmat15 ${BASEINPUT}/scalefree/symmetric/rmat15.sgr -symmetricGraph -kcore=100)

app_dist(kcore_pull)
add_test_dist(kcore_pull rmat15 ${BASEINPUT}/scalefree/symmetric/rmat15.sgr -symmetricGraph -kcore=100)
