/** dGraph loader -*- C++ -*-
 * @file
 * dGraphLoader.h
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Command line arguments and functions for loading dGraphs into memory
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */
#ifndef D_GRAPH_LOADER
#define D_GRAPH_LOADER

#include "galois/graphs/DistributedGraph_EdgeCut.h"
#include "galois/graphs/DistributedGraph_CartesianCut.h"
#include "galois/graphs/DistributedGraph_HybridCut.h"
#include "galois/graphs/DistributedGraph_JaggedCut.h"
#include "galois/graphs/DistributedGraph_CustomEdgeCut.h"

// TODO/FIXME Refactoring a bunch of this code is likely very possible to do
/*******************************************************************************
 * Supported partitioning schemes
 ******************************************************************************/
namespace galois {
namespace graphs {

enum PARTITIONING_SCHEME {
  OEC, IEC, HOVC, HIVC, BOARD2D_VCUT, CART_VCUT, JAGGED_CYCLIC_VCUT,
  JAGGED_BLOCKED_VCUT, OVER_DECOMPOSE_2_VCUT, OVER_DECOMPOSE_4_VCUT,
  CEC
};

inline const char* EnumToString(PARTITIONING_SCHEME e) {
  switch(e) {
    case OEC: return "oec";
    case IEC: return "iec";
    case HOVC: return "hovc";
    case HIVC: return "hivc";
    case BOARD2D_VCUT: return "board2d_vcut";
    case CART_VCUT: return "cvc";
    case JAGGED_CYCLIC_VCUT: return "jcvc";
    case JAGGED_BLOCKED_VCUT: return "jbvc";
    case OVER_DECOMPOSE_2_VCUT: return "od2vc";
    case OVER_DECOMPOSE_4_VCUT: return "od4vc";
    case CEC: return "cec";
    default: GALOIS_DIE("Unsupported partition");
  }
}
} // end namespace graphs
} // end namespace galois

/*******************************************************************************
 * Graph-loading-related command line arguments
 ******************************************************************************/
namespace cll = llvm::cl;

extern cll::opt<std::string> inputFile;
extern cll::opt<std::string> inputFileTranspose;
extern cll::opt<bool> inputFileSymmetric;
extern cll::opt<std::string> partFolder;
extern cll::opt<galois::graphs::PARTITIONING_SCHEME> partitionScheme;
extern cll::opt<unsigned int> VCutThreshold;
extern cll::opt<std::string> vertexIDMapFileName;
extern cll::opt<bool> readFromFile;
extern cll::opt<std::string> localGraphFileName;
extern cll::opt<bool> saveLocalGraph;

namespace galois {
namespace graphs {

/*******************************************************************************
 * Graph-loading functions
 ******************************************************************************/
/**
 * Loads a graph file with the purpose of iterating over the out edges
 * AND in edges of a graph.
 *
 * @tparam NodeData node data to store in graph
 * @tparam EdgeData edge data to store in graph
 *
 * @param scaleFactor How to split nodes among hosts
 * @returns a pointer to a newly allocated DistGraph based on the command line
 * loaded based on command line arguments
 */
template<typename NodeData, typename EdgeData, bool HasNoLockable>
DistGraph<NodeData, EdgeData, true, HasNoLockable>* 
constructTwoWayGraph(std::vector<unsigned>& scaleFactor) {
  // TODO template args for 2-way for everything that isn't edge cut
  // TODO do it for edge cut too
  using Graph_edgeCut = 
      DistGraph_edgeCut<NodeData, EdgeData, true, HasNoLockable>;
  //typedef DistGraph_customEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  //typedef DistGraph_vertexCut<NodeData, EdgeData> Graph_vertexCut;
  //typedef DistGraph_cartesianCut<NodeData, EdgeData> Graph_cartesianCut; // assumes push-style
  //typedef DistGraph_cartesianCut<NodeData, EdgeData, true> 
  //      Graph_checkerboardCut; // assumes push-style
  //typedef DistGraph_jaggedCut<NodeData, EdgeData> 
  //      Graph_jaggedCut; // assumes push-style
  //typedef DistGraph_jaggedCut<NodeData, EdgeData, true> 
  //      Graph_jaggedBlockedCut; // assumes push-style
  //typedef DistGraph_cartesianCut<NodeData, EdgeData, false, false, 2> 
  //                            Graph_cartesianCut_overDecomposeBy2;
  //typedef DistGraph_cartesianCut<NodeData, EdgeData, false, false, 4> 
  //                            Graph_cartesianCut_overDecomposeBy4;

  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut, no transpose
  if (net.Num == 1) {
    return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                             scaleFactor, false, readFromFile, localGraphFileName);
  }

  switch (partitionScheme) {
    case OEC:
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, false, readFromFile, localGraphFileName);
    case IEC:
      if (inputFileTranspose.size()) {
        return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, 
                                 net.Num, scaleFactor, true, readFromFile, 
                                 localGraphFileName);
      } else {
        GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                   "graph");
        break;
      }
    //case HOVC:
    //  return new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
    //                             scaleFactor, false, VCutThreshold, false, 
    //                             readFromFile, localGraphFileName);
    //case HIVC:
    //  if (inputFileTranspose.size()) {
    //    return new Graph_vertexCut(inputFileTranspose, partFolder, net.ID, 
    //                               net.Num, scaleFactor, true, VCutThreshold, false, 
    //                               readFromFile, localGraphFileName);
    //  } else {
    //    GALOIS_DIE("Error: attempting incoming hybrid cut without transpose "
    //               "graph");
    //    break;
    //  }
    //case BOARD2D_VCUT:
    //  return new Graph_checkerboardCut(inputFile, partFolder, net.ID, net.Num, 
    //                                scaleFactor, false);
    //case CART_VCUT:
    //  return new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num, 
    //                                scaleFactor, false,
    //                               readFromFile, localGraphFileName);
    //case JAGGED_CYCLIC_VCUT:
    //  return new Graph_jaggedCut(inputFile, partFolder, net.ID, net.Num, 
    //                                scaleFactor, false);
    //case JAGGED_BLOCKED_VCUT:
    //  return new Graph_jaggedBlockedCut(inputFile, partFolder, net.ID, net.Num, 
    //                                scaleFactor, false);
    //case OVER_DECOMPOSE_2_VCUT:
    //  return new Graph_cartesianCut_overDecomposeBy2(inputFile, partFolder, 
    //                                net.ID, net.Num, scaleFactor, false);
    //case OVER_DECOMPOSE_4_VCUT:
    //  return new Graph_cartesianCut_overDecomposeBy4(inputFile, partFolder, 
    //                                net.ID, net.Num, scaleFactor, false);
    //case CEC:
    //  return new Graph_customEdgeCut(inputFile, partFolder, net.ID, net.Num, 
    //                           scaleFactor, vertexIDMapFileName, false);
    default:
      GALOIS_DIE("Error: partition scheme specified is invalid");
      return nullptr;
  }
}





/**
 * Loads a symmetric graph file (i.e. directed graph with edges in both 
 * directions)
 *
 * @tparam NodeData node data to store in graph
 * @tparam EdgeData edge data to store in graph
 * @param scaleFactor How to split nodes among hosts
 * @returns a pointer to a newly allocated DistGraph based on the command line
 * loaded based on command line arguments
 */
template<typename NodeData, typename EdgeData>
DistGraph<NodeData, EdgeData>* constructSymmetricGraph(std::vector<unsigned>&
                                                    scaleFactor) {
  if (!inputFileSymmetric) {
    GALOIS_DIE("Calling constructSymmetricGraph without inputFileSymmetric "
               "flag");
  }

  typedef DistGraph_edgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef DistGraph_vertexCut<NodeData, EdgeData> Graph_vertexCut;
  typedef DistGraph_cartesianCut<NodeData, EdgeData> Graph_cartesianCut;
  typedef DistGraph_cartesianCut<NodeData, EdgeData, true> Graph_checkerboardCut;
  typedef DistGraph_jaggedCut<NodeData, EdgeData> Graph_jaggedCut;
  typedef DistGraph_jaggedCut<NodeData, EdgeData, true> Graph_jaggedBlockedCut;
  typedef DistGraph_cartesianCut<NodeData, EdgeData, false, false, 2> 
                              Graph_cartesianCut_overDecomposeBy2;
  typedef DistGraph_cartesianCut<NodeData, EdgeData, false, false, 4> 
                              Graph_cartesianCut_overDecomposeBy4;
  typedef DistGraph_customEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  auto& net = galois::runtime::getSystemNetworkInterface();
  
  switch(partitionScheme) {
    case OEC:
    case IEC:
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, false, readFromFile, localGraphFileName);
    case HOVC:
    case HIVC:
      return new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
                                 scaleFactor, false, VCutThreshold, false,
                                 readFromFile, localGraphFileName);
    case BOARD2D_VCUT:
      return new Graph_checkerboardCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false);
    case CART_VCUT:
      return new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false ,
                                 readFromFile, localGraphFileName );
    case JAGGED_CYCLIC_VCUT:
      return new Graph_jaggedCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false);
    case JAGGED_BLOCKED_VCUT:
      return new Graph_jaggedBlockedCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false);
    case OVER_DECOMPOSE_2_VCUT:
      return new Graph_cartesianCut_overDecomposeBy2(inputFile, partFolder, 
                     net.ID, net.Num, scaleFactor, false);
    case OVER_DECOMPOSE_4_VCUT:
      return new Graph_cartesianCut_overDecomposeBy4(inputFile, partFolder, 
                     net.ID, net.Num, scaleFactor, false);
    case CEC:
      return new Graph_customEdgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, vertexIDMapFileName, false);
    default:
      GALOIS_DIE("Error: partition scheme specified is invalid");
      return nullptr;
  }
}

/**
 * Loads a graph file with the purpose of iterating over the out edges
 * of the graph.
 *
 * @tparam NodeData node data to store in graph
 * @tparam EdgeData edge data to store in graph
 * @tparam iterateOut says if you want to iterate over out edges or not; if
 * false, will iterate over in edgse
 * @tparam enable_if this function  will only be enabled if iterateOut is true
 * @param scaleFactor How to split nodes among hosts
 * @returns a pointer to a newly allocated DistGraph based on the command line
 * loaded based on command line arguments
 */
template<typename NodeData, typename EdgeData, bool iterateOut = true,
         typename std::enable_if<iterateOut>::type* = nullptr>
DistGraph<NodeData, EdgeData>* constructGraph(std::vector<unsigned>& 
                                           scaleFactor) {
  typedef DistGraph_edgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef DistGraph_customEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  typedef DistGraph_vertexCut<NodeData, EdgeData> Graph_vertexCut;
  typedef DistGraph_cartesianCut<NodeData, EdgeData> Graph_cartesianCut; // assumes push-style
  typedef DistGraph_cartesianCut<NodeData, EdgeData, true> 
        Graph_checkerboardCut; // assumes push-style
  typedef DistGraph_jaggedCut<NodeData, EdgeData> 
        Graph_jaggedCut; // assumes push-style
  typedef DistGraph_jaggedCut<NodeData, EdgeData, true> 
        Graph_jaggedBlockedCut; // assumes push-style
  typedef DistGraph_cartesianCut<NodeData, EdgeData, false, false, 2> 
                              Graph_cartesianCut_overDecomposeBy2;
  typedef DistGraph_cartesianCut<NodeData, EdgeData, false, false, 4> 
                              Graph_cartesianCut_overDecomposeBy4;

  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut, no transpose
  if (net.Num == 1) {
    return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                             scaleFactor, false, readFromFile, localGraphFileName);
  }

  switch(partitionScheme) {
    case OEC:
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, false, readFromFile, localGraphFileName);
    case IEC:
      if (inputFileTranspose.size()) {
        return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, 
                                 net.Num, scaleFactor, true, readFromFile, localGraphFileName);
      } else {
        GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                   "graph");
        break;
      }
    case HOVC:
      return new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
                                 scaleFactor, false, VCutThreshold, false, 
                                 readFromFile, localGraphFileName);
    case HIVC:
      if (inputFileTranspose.size()) {
        return new Graph_vertexCut(inputFileTranspose, partFolder, net.ID, 
                                   net.Num, scaleFactor, true, VCutThreshold, false, 
                                   readFromFile, localGraphFileName);
      } else {
        GALOIS_DIE("Error: attempting incoming hybrid cut without transpose "
                   "graph");
        break;
      }
    case BOARD2D_VCUT:
      return new Graph_checkerboardCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false);
    case CART_VCUT:
      return new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false,
                                   readFromFile, localGraphFileName);
    case JAGGED_CYCLIC_VCUT:
      return new Graph_jaggedCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false);
    case JAGGED_BLOCKED_VCUT:
      return new Graph_jaggedBlockedCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false);
    case OVER_DECOMPOSE_2_VCUT:
      return new Graph_cartesianCut_overDecomposeBy2(inputFile, partFolder, 
                                    net.ID, net.Num, scaleFactor, false);
    case OVER_DECOMPOSE_4_VCUT:
      return new Graph_cartesianCut_overDecomposeBy4(inputFile, partFolder, 
                                    net.ID, net.Num, scaleFactor, false);
    case CEC:
      return new Graph_customEdgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, vertexIDMapFileName, false);
    default:
      GALOIS_DIE("Error: partition scheme specified is invalid");
      return nullptr;
  }
}

/**
 * Loads a graph file with the purpose of iterating over the in edges
 * of the graph.
 *
 * @tparam NodeData node data to store in graph
 * @tparam EdgeData edge data to store in graph
 * @tparam iterateOut says if you want to iterate over out edges or not; if
 * false, will iterate over in edgse
 * @tparam enable_if this function  will only be enabled if iterateOut is false
 * (i.e. iterate over in-edges)
 * @param scaleFactor How to split nodes among hosts
 * @returns a pointer to a newly allocated DistGraph based on the command line
 * loaded based on command line arguments
 */
template<typename NodeData, typename EdgeData, bool iterateOut = true,
         typename std::enable_if<!iterateOut>::type* = nullptr>
DistGraph<NodeData, EdgeData>* constructGraph(std::vector<unsigned>& scaleFactor) {
  typedef DistGraph_edgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef DistGraph_vertexCut<NodeData, EdgeData> Graph_vertexCut;
  typedef DistGraph_cartesianCut<NodeData, EdgeData, false, 
                              true> Graph_cartesianCut; // assumes pull-style
  typedef DistGraph_cartesianCut<NodeData, EdgeData, true, 
                              true> Graph_checkerboardCut; // assumes pull-style
  typedef DistGraph_jaggedCut<NodeData, EdgeData, false, 
                           true> Graph_jaggedCut; // assumes pull-style
  typedef DistGraph_jaggedCut<NodeData, EdgeData, true, 
                           true> Graph_jaggedBlockedCut; // assumes pull-style
  typedef DistGraph_cartesianCut<NodeData, EdgeData, false, true, 2> 
                              Graph_cartesianCut_overDecomposeBy2; // assumes pull-style
  typedef DistGraph_cartesianCut<NodeData, EdgeData, false, true, 4> 
                              Graph_cartesianCut_overDecomposeBy4; // assumes pull-style
  typedef DistGraph_customEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut
  if (net.Num == 1) {
    if (inputFileTranspose.size()) {
      return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                               scaleFactor, false, readFromFile, localGraphFileName);
    } else {
      fprintf(stderr, "WARNING: Loading transpose graph through in-memory "
                      "transpose to iterate over in-edges: pass in transpose "
                      "graph with -graphTranspose to avoid unnecessary "
                      "overhead.\n");
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, true, readFromFile, localGraphFileName);
    }
  }


  switch(partitionScheme) {
    case OEC:
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, true, readFromFile, localGraphFileName);
    case IEC:
      if (inputFileTranspose.size()) {
        return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                                 scaleFactor, false, readFromFile, localGraphFileName);
      } else {
        GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                   "graph");
        break;
      }
    case HOVC:
      return new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
                                 scaleFactor, true, VCutThreshold, false, 
                                 readFromFile, localGraphFileName);
    case HIVC:
      if (inputFileTranspose.size()) {
        return new Graph_vertexCut(inputFileTranspose, partFolder, net.ID, 
                                   net.Num, scaleFactor, false, VCutThreshold, false, 
                                   readFromFile, localGraphFileName);
      } else {
        GALOIS_DIE("Error: (hivc) iterate over in-edges without transpose graph");
        break;
      }

    case BOARD2D_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_checkerboardCut(inputFileTranspose, partFolder, 
                                         net.ID, net.Num, scaleFactor, false);
      } else {
        GALOIS_DIE("Error: (cvc) iterate over in-edges without transpose graph");
        break;
      }

    case CART_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_cartesianCut(inputFileTranspose, partFolder, net.ID, 
                                      net.Num, scaleFactor, false, 
                                   readFromFile, localGraphFileName);
      } else {
        GALOIS_DIE("Error: (cvc) iterate over in-edges without transpose graph");
        break;
      }
    case JAGGED_CYCLIC_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_jaggedCut(inputFileTranspose, partFolder, net.ID, 
                                   net.Num, scaleFactor, false);
      } else {
        GALOIS_DIE("Error: (jcvc) iterate over in-edges without transpose graph");
        break;
      }
    case JAGGED_BLOCKED_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_jaggedBlockedCut(inputFileTranspose, partFolder, 
                                          net.ID, net.Num, scaleFactor, false);
      } else {
        GALOIS_DIE("Error: (jbvc) iterate over in-edges without transpose graph");
        break;
      }
    case OVER_DECOMPOSE_2_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_cartesianCut_overDecomposeBy2(inputFileTranspose, 
                         partFolder, net.ID, net.Num, scaleFactor, false);
      } else {
        GALOIS_DIE("Error: (od2vc) iterate over in-edges without transpose graph");
        break;
      }
    case OVER_DECOMPOSE_4_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_cartesianCut_overDecomposeBy4(inputFileTranspose, 
                         partFolder, net.ID, net.Num, scaleFactor, false);
      } else {
        GALOIS_DIE("Error: (od4vc) iterate over in-edges without transpose graph");
        break;
      }

    case CEC:
      if (inputFileTranspose.size()) {
        return new Graph_customEdgeCut(inputFileTranspose, partFolder, net.ID, 
                                       net.Num, scaleFactor, vertexIDMapFileName, 
                                       false);
      } else {
        GALOIS_DIE("Error: (od4vc) iterate over in-edges without transpose graph");
        break;
      }
    default:
      GALOIS_DIE("Error: partition scheme specified is invalid");
      return nullptr;
  }
}

} // end namespace graphs
} // end namespace galois
#endif
