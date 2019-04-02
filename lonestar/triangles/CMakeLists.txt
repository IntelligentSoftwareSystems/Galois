app(triangles Triangles.cpp)
app(trianglesAOS Triangles_AOS.cpp)

add_test_scale(node-small triangles -algo=nodeiterator "${BASEINPUT}/stanford/communities/DBLP/com-dblp.wgt32.sym.gr.triangles")
add_test_scale(edge-small triangles -algo=edgeiterator "${BASEINPUT}/stanford/communities/DBLP/com-dblp.wgt32.sym.gr.triangles")
