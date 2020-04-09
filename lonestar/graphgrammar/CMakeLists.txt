app(graphgrammar src/GraphGrammar.cpp src/model/Map.cpp src/readers/SrtmReader.cpp src/readers/AsciiReader.cpp
        src/libmgrs/mgrs.c src/libmgrs/polarst.c src/libmgrs/tranmerc.c src/libmgrs/utm.c src/libmgrs/ups.c
        src/utils/Utils.cpp src/readers/InpReader.cpp src/writers/InpWriter.cpp)

app(graphgrammarTest test/TestMain.cpp src/model/Map.cpp src/libmgrs/mgrs.c src/libmgrs/polarst.c src/libmgrs/tranmerc.c
        src/libmgrs/utm.c src/libmgrs/ups.c src/utils/Utils.cpp)
