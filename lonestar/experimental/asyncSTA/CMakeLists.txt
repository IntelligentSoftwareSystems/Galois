if(USE_EXP)
  include_directories(${CMAKE_SOURCE_DIR}/lonestar/experimental/sta .)
endif()
app(asyncSTA AsyncSTA.cpp AsyncTimingGraph.cpp AsyncTimingEngine.cpp AsyncTimingArcSet.cpp ../sta/TimingDefinition.cpp ../sta/WireLoad.cpp ../sta/CellLib.cpp ../sta/Verilog.cpp ../sta/Tokenizer.cpp)
