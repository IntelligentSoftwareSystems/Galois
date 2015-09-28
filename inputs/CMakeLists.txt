file(DOWNLOAD http://iss.ices.utexas.edu/projects/galois/downloads/lonestar-cpu-inputs.tar.xz ${CMAKE_CURRENT_BINARY_DIR}/lonestar-cpu-inputs.tar.xz SHOW_PROGRESS)

set(INPUT_GRAPHS
        avi/10x10_42k.NEU.gz
        scalefree/rmat8-2e14.gr
        structured/torus5.gr
        structured/rome99.gr
        random/r4-2e23.gr
        random/r4-2e24.gr
        random/r4-2e26.gr
        road/USA-road-d.USA.gr
        meshes/r5M.ele
        meshes/r5M.node
        meshes/r5M.points
        meshes/r5M.poly
        meshes/r10k.1.ele
        meshes/r10k.1.node
        meshes/r10k.1.poly
        meshes/r10k.node
        des/multTree6bit.net
        des/koggeStone64bit.net
)

add_custom_command(
OUTPUT ${INPUT_GRAPHS}
COMMAND ${CMAKE_COMMAND} -E tar xJf lonestar-cpu-inputs.tar.xz
DEPENDS lonestar-cpu-inputs.tar.xz
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/inputs
COMMENT "Unpacking lonestar-cpu-inputs.tar.xz"
VERBATIM
)
