if(("${CMAKE_VERSION}" VERSION_GREATER 3.8.0) OR ("${CMAKE_VERSION}" VERSION_EQUAL 3.8.0))
    app(ilu ilu.cpp)
    set_property(TARGET ilu PROPERTY CXX_STANDARD 17)
else()
    message(WARNING "Cannot configure ilu or other C++17 apps with a CMake version older than 3.8. Skipping.")
endif()

