add_executable(longestedge-cpu src/LongestEdge.cpp src/model/Map.cpp
               src/readers/SrtmReader.cpp src/readers/AsciiReader.cpp
               src/libmgrs/mgrs.c src/libmgrs/polarst.c src/libmgrs/tranmerc.c
               src/libmgrs/utm.c src/libmgrs/ups.c src/utils/Utils.cpp
               src/readers/InpReader.cpp src/writers/InpWriter.cpp
               src/writers/TriangleFormatWriter.cpp)
add_dependencies(apps longestedge-cpu)
target_link_libraries(longestedge-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS longestedge-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_executable(longestedgeTest test/TestMain.cpp src/model/Map.cpp
               src/libmgrs/mgrs.c src/libmgrs/polarst.c src/libmgrs/tranmerc.c
               src/libmgrs/utm.c src/libmgrs/ups.c src/utils/Utils.cpp)
add_dependencies(apps longestedgeTest)
target_link_libraries(longestedgeTest PRIVATE Galois::shmem lonestar)

add_test(test_longest_edge longestedgeTest)
