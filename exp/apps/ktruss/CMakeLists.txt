app(k-truss K-Truss.cpp)
app(verifyKTruss Verify.cpp)
add_custom_command(TARGET k-truss
                   POST_BUILD
                   COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/bmktest2.py bmktest2.py
		   )
