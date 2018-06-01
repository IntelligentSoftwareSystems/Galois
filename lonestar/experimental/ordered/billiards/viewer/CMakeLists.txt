set(QT_USE_QTOPENGL TRUE)
set(QT_USE_QTXML TRUE)
find_package(Qt4)

find_package(OpenGL)
if(OPENGL_FOUND)
  include_directories(OPENGL_INCLUDE_DIR)
endif()

#QT4_WRAP_CPP(viewer_HEADERS_MOC ...)
set(viewer_HEADERS_MOC)

include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

find_package(QGLViewer)
app(billiards-viewer 
  main.cpp Viewer.cpp Scene.cpp ${viewer_HEADERS_MOC}
  REQUIRES QGLView_FOUND QT4_FOUND
  EXTLIBS ${OPENGL_LIBRARIES} ${QT_LIBRARIES} ${QGLViewer_LIBRARIES})
