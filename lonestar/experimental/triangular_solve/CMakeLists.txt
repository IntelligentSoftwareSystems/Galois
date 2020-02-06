if(("${CMAKE_VERSION}" VERSION_GREATER 3.8.0) OR ("${CMAKE_VERSION}" VERSION_EQUAL 3.8.0))
    app(triangular_solve triangular_solve.cpp EXP_OPT)
    set_property(TARGET triangular_solve PROPERTY CXX_STANDARD 17)
else()
    message(WARNING "Cannot configure triangular_solve or other C++17 apps with a CMake version older than 3.8. Skipping.")
endif()

