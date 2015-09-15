set (Sources dependTest.cpp Ball.cpp Table.cpp Cushion.cpp Sector.cpp FPutils.cpp GeomUtils.cpp) 

app (BilliardsSerialPQ BilliardsSerialPQ.cpp ${Sources})
app (BilliardsSerialSectored BilliardsSerialSectored.cpp ${Sources})
app (BilliardsSerialLog BilliardsSerialLog.cpp ${Sources})
app (BilliardsFromLog BilliardsFromLog.cpp ${Sources})

app (BilliardsPOunsortedSerial BilliardsPOunsortedSerial.cpp ${Sources})
app (BilliardsPOsortedSerial BilliardsPOsortedSerial.cpp ${Sources})

app (BilliardsPOunsorted BilliardsPOunsorted.cpp ${Sources})
app (BilliardsPOsortedVec BilliardsPOsortedVec.cpp ${Sources})
app (BilliardsLevelExec BilliardsLevelExec.cpp ${Sources})
app (BilliardsTwoPhase BilliardsTwoPhase.cpp ${Sources})
app (BilliardsSpec BilliardsSpec.cpp ${Sources})
app (BilliardsSectoredIKDG BilliardsSectoredIKDG.cpp ${Sources})
add_subdirectory(viewer)
