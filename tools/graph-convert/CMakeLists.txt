app(graph-convert graph-convert.cpp DISTSAFE)
install(TARGETS graph-convert EXPORT GaloisTargets RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
