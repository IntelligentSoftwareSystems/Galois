configure_file("make_dist.sh.in" "make_dist.sh")
file(COPY .  DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PATTERN .svn EXCLUDE)
