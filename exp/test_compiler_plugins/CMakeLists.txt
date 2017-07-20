#Test loop transform plugin.
app(loopTransform loopTransform.cpp DISTSAFE)


#Test simple push version
app(sssp_push sssp_push.cpp DISTSAFE)


#Test simple push version
app(cc_pull cc_pull.cpp DISTSAFE)


#Test simple push wl version
app(bfs_wl bfs_wl.cpp DISTSAFE)
