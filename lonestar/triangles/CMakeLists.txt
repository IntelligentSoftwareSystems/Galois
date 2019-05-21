app(triangles Triangles.cpp)
app(trianglesAOS Triangles_AOS.cpp)

add_test_scale(small-node triangles -algo=nodeiterator "${BASEINPUT}/stanford/communities/DBLP/com-dblp.wgt32.sym.gr.triangles")
add_test_scale(small-edge triangles -algo=edgeiterator "${BASEINPUT}/stanford/communities/DBLP/com-dblp.wgt32.sym.gr.triangles")
