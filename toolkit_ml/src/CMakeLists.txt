add_library(toolkit_ml svm.cpp)
target_link_libraries(toolkit_ml galois)

# Required for python build
set_target_properties(toolkit_ml PROPERTIES POSITION_INDEPENDENT_CODE True)
