app(bp bp.cpp)

add_definitions(-Wno-deprecated)
add_definitions(-DDAI_WITH_BP -DDAI_WITH_FBP -DDAI_WITH_TRWBP -DDAI_WITH_MF -DDAI_WITH_HAK)
add_definitions(-DDAI_WITH_LC -DDAI_WITH_TREEEP -DDAI_WITH_JTREE -DDAI_WITH_MR -DDAI_WITH_GIBBS)
add_definitions(-DDAI_WITH_CBP -DDAI_WITH_DECMAP)
file(GLOB Sources libdai/src/*.cpp)
include_directories("libdai/include")
app(bp-dai bp-dai.cpp ${Sources} EXTLIBS gmp gmpxx)
