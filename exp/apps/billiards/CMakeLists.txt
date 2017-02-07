set (Sources dependTest.cpp DefaultValues.cpp Ball.cpp Cushion.cpp Sector.cpp FPutils.cpp GeomUtils.cpp) 

app (BilliardsSerialPQ BilliardsSerialPQ.cpp ${Sources} EXP_OPT)
app (BilliardsSerialGalloc BilliardsSerialGalloc.cpp ${Sources} EXP_OPT)
app (BilliardsSerialSectored BilliardsSerialSectored.cpp ${Sources} EXP_OPT)
app (BilliardsSerialLog BilliardsSerialLog.cpp ${Sources} EXP_OPT)
app (BilliardsFromLog BilliardsFromLog.cpp ${Sources} EXP_OPT)

app (BilliardsPOunsortedSerial BilliardsPOunsortedSerial.cpp ${Sources} EXP_OPT)
app (BilliardsPOsortedSerial BilliardsPOsortedSerial.cpp ${Sources} EXP_OPT)

app (BilliardsPOunsorted BilliardsPOunsorted.cpp ${Sources} EXP_OPT)
app (BilliardsPOsortedVec BilliardsPOsortedVec.cpp ${Sources} EXP_OPT)
app (BilliardsLevelExec BilliardsLevelExec.cpp ${Sources} EXP_OPT)
app (BilliardsTwoPhase BilliardsTwoPhase.cpp ${Sources} EXP_OPT)
app (BilliardsSpec BilliardsSpec.cpp ${Sources} EXP_OPT)
app (BilliardsOptim BilliardsOptim.cpp ${Sources} EXP_OPT)
app (BilliardsSectoredIKDG BilliardsSectoredIKDG.cpp ${Sources} EXP_OPT)
# add_subdirectory(viewer)
