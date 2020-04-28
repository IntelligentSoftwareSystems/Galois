cmake_minimum_required(VERSION 3.13)

project(pygalois)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

add_subdirectory(.. ${CMAKE_BINARY_DIR}/../cmake-galois-build EXCLUDE_FROM_ALL)

add_subdirectory(galois)
