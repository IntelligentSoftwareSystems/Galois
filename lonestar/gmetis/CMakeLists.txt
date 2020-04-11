app(gmetis Coarsening.cpp GMetis.cpp Metric.cpp Partitioning.cpp Refine.cpp)
# Disable failing test (issue #116).
add_test_scale(small1 gmetis "${BASEINPUT}/reference/structured/rome99.gr" 4 NOT_QUICK)
add_test_scale(small2 gmetis "${BASEINPUT}/scalefree/rmat10.gr" 256)
