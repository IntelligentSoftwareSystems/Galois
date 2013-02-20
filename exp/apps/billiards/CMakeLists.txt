set (Sources dependTest.cpp Table.cpp Cushion.cpp FPutils.cpp Vec2.cpp Event.cpp)
app (BilliardsSerialPQ BilliardsSerialPQ.cpp ${Sources})
app (BilliardsSerialLog BilliardsSerialLog.cpp ${Sources})
app (BilliardsFromLog BilliardsFromLog.cpp ${Sources})

app (BilliardsPOunsortedSerial BilliardsPOunsortedSerial.cpp ${Sources})
app (BilliardsPOsortedSerial BilliardsPOsortedSerial.cpp ${Sources})

app (BilliardsPOunsorted BilliardsPOunsorted.cpp ${Sources})
app (BilliardsPOsortedVec BilliardsPOsortedVec.cpp ${Sources})

find_package(QGLViewer)
if (QGLVIEWER_FOUND)
  add_subdirectory(viewer)
endif()
