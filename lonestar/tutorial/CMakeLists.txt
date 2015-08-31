app(hello-world HelloWorld.cpp)
app(torus Torus.cpp)
app(torus-improved TorusImproved.cpp)
app(sssp-simple SSSPsimple.cpp)
app(sssp-pull-simple SSSPPullSimple.cpp)

add_test_scale(small hello-world 2 10)
add_test_scale(small torus 2 100)
add_test_scale(small torus-improved 2 100)
