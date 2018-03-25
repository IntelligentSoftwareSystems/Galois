app(surveypropagation SurveyPropagation.cpp)
app(surveypropagation_SOA SurveyPropagation_SOA.cpp)

add_test_scale(small surveypropagation 9 100 300 3)
add_test_scale(web surveypropagation 9 1000000 3000000 3)
