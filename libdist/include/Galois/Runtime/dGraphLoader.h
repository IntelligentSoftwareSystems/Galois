#ifndef D_GRAPH_LOADER
#define D_GRAPH_LOADER

#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/dGraph_edgeCut.h"
#include "Galois/Runtime/dGraph_cartesianCut.h"
#include "Galois/Runtime/dGraph_hybridCut.h"

enum PARTITIONING_SCHEME {
  OEC, IEC, PL_VCUT, CART_VCUT
};
//enum EDGE_ITERATE { ITERATE_OUT, ITERATE_IN };

namespace cll = llvm::cl;

static cll::opt<std::string> inputFile(cll::Positional, 
                                       cll::desc("<input file>"), 
                                       cll::Required);
static cll::opt<std::string> inputFileTranspose("graphTranspose",
                                       cll::desc("<input file, transposed>"), 
                                       cll::init(""));
static cll::opt<bool> inputFileSymmetric("symmetricGraph",
                                       cll::desc("Set this flag if graph is symmetric"), 
                                       cll::init(false));
static cll::opt<std::string> partFolder("partFolder", 
                                        cll::desc("path to partitionFolder"), 
                                        cll::init(""));
static cll::opt<PARTITIONING_SCHEME> partitionScheme("partition",
                                     cll::desc("Type of partitioning."),
                                     cll::values(
                                       clEnumValN(OEC, "oec", 
                                                  "Outgoing edge cut"), 
                                       clEnumValN(IEC, "iec", 
                                                  "Incoming edge cut"), 
                                       clEnumValN(PL_VCUT, "pl_vcut", 
                                                  "Powerlyra Vertex Cut"), 
                                       clEnumValN(CART_VCUT , "cart_vcut", 
                                                  "Cartesian Vertex Cut"), 
                                       clEnumValEnd),
                                     cll::init(OEC));
static cll::opt<unsigned int> VCutThreshold("VCutThreshold", 
                                            cll::desc("Threshold for high "
                                                      "degree edges."), 
                                            cll::init(1000));


template<typename NodeData, typename EdgeData, bool iterateOut = true,
         typename std::enable_if<iterateOut>::type* = nullptr>
hGraph<NodeData, EdgeData>* constructGraph(std::vector<unsigned> scaleFactor) {
  typedef hGraph_edgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef hGraph_vertexCut<NodeData, EdgeData> Graph_vertexCut;
  typedef hGraph_cartesianCut<NodeData, EdgeData> Graph_cartesianCut;

  auto& net = Galois::Runtime::getSystemNetworkInterface();
  
  switch(partitionScheme) {
    case OEC:
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, false);
    case IEC:
      if (inputFileTranspose.size()) {
        return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                                 scaleFactor, true);
      } else {
        GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                   "graph");
        break;
      }
    case PL_VCUT:
      return new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
                                 scaleFactor, false, VCutThreshold);
    case CART_VCUT:
      return new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false);
    default:
      GALOIS_DIE("Error: partition scheme specified is invalid");
      return nullptr;
}
}

template<typename NodeData, typename EdgeData, bool iterateOut = true,
         typename std::enable_if<!iterateOut>::type* = nullptr>
hGraph<NodeData, EdgeData>* constructGraph(std::vector<unsigned> scaleFactor) {
  typedef hGraph_edgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef hGraph_vertexCut<NodeData, EdgeData> Graph_vertexCut;
  typedef hGraph_cartesianCut<NodeData, EdgeData> Graph_cartesianCut;

  auto& net = Galois::Runtime::getSystemNetworkInterface();
  
  switch(partitionScheme) {
    case OEC:
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, true);
    case IEC:
      if (inputFileTranspose.size()) {
        return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                                 scaleFactor, false);
      } else {
        GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                   "graph");
        break;
      }
    case PL_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_vertexCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                                   scaleFactor, false, VCutThreshold);
      } else {
        GALOIS_DIE("Error: iterate over in edges without transpose graph");
        break;
      }

    case CART_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_cartesianCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                                      scaleFactor, false);
      } else {
        GALOIS_DIE("Error: iterate over in edges without transpose graph");
        break;
      }
    default:
      GALOIS_DIE("Error: partition scheme specified is invalid");
      return nullptr;
  }
}
#endif
