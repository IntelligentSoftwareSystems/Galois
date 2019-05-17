distApp(kcore_push)
testDistApp(kcore_push rmat15 ${BASEINPUT}/scalefree/rmat15.sgr -symmetricGraph -kcore=100)

distApp(kcore_pull)
testDistApp(kcore_pull rmat15 ${BASEINPUT}/scalefree/rmat15.sgr -symmetricGraph -kcore=100)
