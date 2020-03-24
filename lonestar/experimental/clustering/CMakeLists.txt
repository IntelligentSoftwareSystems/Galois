app(clustering BVHclustering.cpp LightClustering.cpp)

add_test_scale(web clustering -numPoints 10000)
add_test_scale(small clustering -numPoints 1000)
