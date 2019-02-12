/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

/**
 * @file DistributedGraphLoader.h
 *
 * Contains definitions for the common distributed graph loading functionality
 * of Galois.
 *
 * @todo Refactoring a bunch of this code is likely very possible to do
 */
#ifndef D_GRAPH_LOADER
#define D_GRAPH_LOADER

#include "galois/graphs/DistributedGraph_EdgeCut.h"
#include "galois/graphs/DistributedGraph_CartesianCut.h"
#include "galois/graphs/DistributedGraph_CartesianCutOld.h"
#include "galois/graphs/DistributedGraph_HybridCut.h"
#include "galois/graphs/DistributedGraph_JaggedCut.h"
#include "galois/graphs/DistributedGraph_CustomEdgeCut.h"
#include "galois/graphs/Generic.h"
#include "galois/graphs/NewGeneric.h"
#include "galois/graphs/GenericPartitioners.h"

/*******************************************************************************
 * Supported partitioning schemes
 ******************************************************************************/
namespace galois {
namespace graphs {

//! enums of partitioning schemes supported
enum PARTITIONING_SCHEME {
  OEC,                   //!< outgoing edge cut
  IEC,                   //!< incoming edge cut
  HOVC,                  //!< outgoing hybrid vertex cut
  HIVC,                  //!< incoming hybrid vertex cut
  BOARD2D_VCUT,          //!< checkerboard cut
  CART_VCUT,             //!< cartesian vertex cut
  CART_VCUT_IEC,         //!< cartesian vertex cut using iec
  CART_VCUT_OLD,             //!< cartesian vertex cut
  JAGGED_CYCLIC_VCUT,    //!< cyclic jagged cut
  JAGGED_BLOCKED_VCUT,   //!< blocked jagged cut
  OVER_DECOMPOSE_2_VCUT, //!< overdecompose cvc by 2
  OVER_DECOMPOSE_4_VCUT, //!< overdecompose cvc by 4
  CEC,                    //!< custom edge cut
  GCVC,                    //!< generic cvc
  GHIVC,                    //!< generic hivc
  GOEC,                    //!< generic oec
  GING,                    //!< Ginger
  FENNEL_O,                   //!< Fennel, oec
  FENNEL_I                    //!< Fennel, iec
};

/**
 * Turns a PARTITIONING_SCHEME enum to a string
 *
 * @param e partitioning scheme enum
 * @return string version of e
 */
inline const char* EnumToString(PARTITIONING_SCHEME e) {
  switch (e) {
  case OEC:
    return "oec";
  case IEC:
    return "iec";
  case HOVC:
    return "hovc";
  case HIVC:
    return "hivc";
  case BOARD2D_VCUT:
    return "board2d_vcut";
  case CART_VCUT:
    return "cvc";
  case CART_VCUT_IEC:
    return "cvc_iec";
  case CART_VCUT_OLD:
    return "cvc_old";
  case JAGGED_CYCLIC_VCUT:
    return "jcvc";
  case JAGGED_BLOCKED_VCUT:
    return "jbvc";
  case OVER_DECOMPOSE_2_VCUT:
    return "od2vc";
  case OVER_DECOMPOSE_4_VCUT:
    return "od4vc";
  case CEC:
    return "cec";
  case GCVC:
    return "gcvc";
  case GHIVC:
    return "ghivc";
  case GOEC:
    return "goec";
  case GING:
    return "ginger";
  case FENNEL_O:
    return "fennel-oec";
  case FENNEL_I:
    return "fennel-iec";
  default:
    GALOIS_DIE("Unsupported partition");
  }
}
} // end namespace graphs
} // end namespace galois

/*******************************************************************************
 * Graph-loading-related command line arguments
 ******************************************************************************/
namespace cll = llvm::cl;

//! input graph file
extern cll::opt<std::string> inputFile;
//! input graph file, but transposed
extern cll::opt<std::string> inputFileTranspose;
//! symmetric input graph file
extern cll::opt<bool> inputFileSymmetric;
//! partition folder; unused
//! @deprecated Not being used anymore; needs to be removed in the future
extern cll::opt<std::string> partFolder;
//! partitioning scheme to use
extern cll::opt<galois::graphs::PARTITIONING_SCHEME> partitionScheme;
//! threshold to determine how hybrid vertex cut assigned edges
extern cll::opt<unsigned int> VCutThreshold;
//! path to vertex id map for custom edge cut
extern cll::opt<std::string> vertexIDMapFileName;
//! true if you want to read graph structure from a file
extern cll::opt<bool> readFromFile;
//! path to local graph structure to read
extern cll::opt<std::string> localGraphFileName;
//! if true, the local graph structure will be saved to disk after partitioning
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
 *
 * @todo Add support for other cuts besides edge cut
 */
template <typename NodeData, typename EdgeData>
DistGraph<NodeData, EdgeData, true>*
constructTwoWayGraph(std::vector<unsigned>& scaleFactor) {
  // TODO template args for 2-way for everything that isn't edge cut
  // TODO do it for edge cut too
  using Graph_edgeCut = DistGraphEdgeCut<NodeData, EdgeData, true>;
  // typedef DistGraphCustomEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  // typedef DistGraphHybridCut<NodeData, EdgeData> Graph_vertexCut;
  // typedef DistGraphCartesianCut<NodeData, EdgeData> Graph_cartesianCut; //
  // assumes push-style typedef DistGraphCartesianCut<NodeData, EdgeData, true>
  //      Graph_checkerboardCut; // assumes push-style
  // typedef DistGraphJaggedCut<NodeData, EdgeData>
  //      Graph_jaggedCut; // assumes push-style
  // typedef DistGraphJaggedCut<NodeData, EdgeData, true>
  //      Graph_jaggedBlockedCut; // assumes push-style
  // typedef DistGraphCartesianCut<NodeData, EdgeData, false, false, 2>
  //                            Graph_cartesianCut_overDecomposeBy2;
  // typedef DistGraphCartesianCut<NodeData, EdgeData, false, false, 4>
  //                            Graph_cartesianCut_overDecomposeBy4;

  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut, no transpose
  if (net.Num == 1) {
    return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                             scaleFactor, false, readFromFile,
                             localGraphFileName);
  }

  switch (partitionScheme) {
  case OEC:
    return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                             scaleFactor, false, readFromFile,
                             localGraphFileName);
  case IEC:
    if (inputFileTranspose.size()) {
      return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num,
                               scaleFactor, true, readFromFile,
                               localGraphFileName);
    } else {
      GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                 "graph");
      break;
    }
  // case HOVC:
  //  return new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num,
  //                             scaleFactor, false, VCutThreshold, false,
  //                             readFromFile, localGraphFileName);
  // case HIVC:
  //  if (inputFileTranspose.size()) {
  //    return new Graph_vertexCut(inputFileTranspose, partFolder, net.ID,
  //                               net.Num, scaleFactor, true, VCutThreshold,
  //                               false, readFromFile, localGraphFileName);
  //  } else {
  //    GALOIS_DIE("Error: attempting incoming hybrid cut without transpose "
  //               "graph");
  //    break;
  //  }
  // case BOARD2D_VCUT:
  //  return new Graph_checkerboardCut(inputFile, partFolder, net.ID, net.Num,
  //                                scaleFactor, false);
  // case CART_VCUT:
  //  return new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num,
  //                                scaleFactor, false,
  //                               readFromFile, localGraphFileName);
  // case CART_VCUT_IEC:
  //  return new Graph_cartesianCut(inputFileTranspose, partFolder, net.ID, net.Num,
  //                                scaleFactor, true,
  //                               readFromFile, localGraphFileName);
  // case JAGGED_CYCLIC_VCUT:
  //  return new Graph_jaggedCut(inputFile, partFolder, net.ID, net.Num,
  //                                scaleFactor, false);
  // case JAGGED_BLOCKED_VCUT:
  //  return new Graph_jaggedBlockedCut(inputFile, partFolder, net.ID, net.Num,
  //                                scaleFactor, false);
  // case OVER_DECOMPOSE_2_VCUT:
  //  return new Graph_cartesianCut_overDecomposeBy2(inputFile, partFolder,
  //                                net.ID, net.Num, scaleFactor, false);
  // case OVER_DECOMPOSE_4_VCUT:
  //  return new Graph_cartesianCut_overDecomposeBy4(inputFile, partFolder,
  //                                net.ID, net.Num, scaleFactor, false);
  // case CEC:
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
template <typename NodeData, typename EdgeData>
DistGraph<NodeData, EdgeData>*
constructSymmetricGraph(std::vector<unsigned>& scaleFactor) {
  if (!inputFileSymmetric) {
    GALOIS_DIE("Calling constructSymmetricGraph without inputFileSymmetric "
               "flag");
  }

  typedef DistGraphEdgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef DistGraphHybridCut<NodeData, EdgeData> Graph_vertexCut;
  typedef DistGraphCartesianCut<NodeData, EdgeData> Graph_cartesianCut;
  typedef DistGraphCartesianCutOld<NodeData, EdgeData> Graph_cartesianCutOld;
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, true> Graph_checkerboardCut;
  typedef DistGraphJaggedCut<NodeData, EdgeData> Graph_jaggedCut;
  typedef DistGraphJaggedCut<NodeData, EdgeData, true> Graph_jaggedBlockedCut;
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, false, false, 2>
      Graph_cartesianCut_overDecomposeBy2;
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, false, false, 4>
      Graph_cartesianCut_overDecomposeBy4;
  typedef DistGraphCustomEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  using GenericCVC = DistGraphGeneric<NodeData, EdgeData, GenericCVC>;
  using GenericHVC = DistGraphGeneric<NodeData, EdgeData, GenericHVC>;
  using GenericEC = DistGraphGeneric<NodeData, EdgeData, NoCommunication>;
  using Ging = NewDistGraphGeneric<NodeData, EdgeData, GingerP>;
  using Fenn = NewDistGraphGeneric<NodeData, EdgeData, FennelP>;

  auto& net = galois::runtime::getSystemNetworkInterface();

  switch (partitionScheme) {
  case OEC:
  case IEC:
    return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                             scaleFactor, false, readFromFile,
                             localGraphFileName);
  case HOVC:
  case HIVC:
    return new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num,
                               scaleFactor, false, VCutThreshold, false,
                               readFromFile, localGraphFileName);
  case BOARD2D_VCUT:
    return new Graph_checkerboardCut(inputFile, partFolder, net.ID, net.Num,
                                     scaleFactor, false);
  case CART_VCUT:
  case CART_VCUT_IEC:
    return new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num,
                                  scaleFactor, false, readFromFile,
                                  localGraphFileName);
  case CART_VCUT_OLD:
    return new Graph_cartesianCutOld(inputFile, partFolder, net.ID, net.Num,
                                     scaleFactor, false, readFromFile,
                                     localGraphFileName);
  case JAGGED_CYCLIC_VCUT:
    return new Graph_jaggedCut(inputFile, partFolder, net.ID, net.Num,
                               scaleFactor, false);
  case JAGGED_BLOCKED_VCUT:
    return new Graph_jaggedBlockedCut(inputFile, partFolder, net.ID, net.Num,
                                      scaleFactor, false);
  case OVER_DECOMPOSE_2_VCUT:
    return new Graph_cartesianCut_overDecomposeBy2(
        inputFile, partFolder, net.ID, net.Num, scaleFactor, false);
  case OVER_DECOMPOSE_4_VCUT:
    return new Graph_cartesianCut_overDecomposeBy4(
        inputFile, partFolder, net.ID, net.Num, scaleFactor, false);
  case CEC:
    return new Graph_customEdgeCut(inputFile, partFolder, net.ID, net.Num,
                                   scaleFactor, vertexIDMapFileName, false);
  case GCVC:
    return new GenericCVC(inputFile, net.ID, net.Num, false,
                          readFromFile, localGraphFileName);
  case GHIVC:
    return new GenericHVC(inputFile, net.ID, net.Num, false);

  case GOEC:
    return new GenericEC(inputFile, net.ID, net.Num, false);

  case GING:
    return new Ging(inputFile, net.ID, net.Num, false);

  case FENNEL_O:
    return new Fenn(inputFile, net.ID, net.Num, false);

  case FENNEL_I:
    return new Fenn(inputFile, net.ID, net.Num, false);

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
template <typename NodeData, typename EdgeData, bool iterateOut = true,
          typename std::enable_if<iterateOut>::type* = nullptr>
DistGraph<NodeData, EdgeData>*
constructGraph(std::vector<unsigned>& scaleFactor) {
  typedef DistGraphEdgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef DistGraphCustomEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  typedef DistGraphHybridCut<NodeData, EdgeData> Graph_vertexCut;
  typedef DistGraphCartesianCut<NodeData, EdgeData>
      Graph_cartesianCut; // assumes push-style
  typedef DistGraphCartesianCutOld<NodeData, EdgeData>
      Graph_cartesianCutOld;
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, true>
      Graph_checkerboardCut; // assumes push-style
  typedef DistGraphJaggedCut<NodeData, EdgeData>
      Graph_jaggedCut; // assumes push-style
  typedef DistGraphJaggedCut<NodeData, EdgeData, true>
      Graph_jaggedBlockedCut; // assumes push-style
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, false, false, 2>
      Graph_cartesianCut_overDecomposeBy2;
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, false, false, 4>
      Graph_cartesianCut_overDecomposeBy4;
  using GenericCVC = DistGraphGeneric<NodeData, EdgeData, GenericCVC>;
  using GenericHVC = DistGraphGeneric<NodeData, EdgeData, GenericHVC>;
  using GenericEC = DistGraphGeneric<NodeData, EdgeData, NoCommunication>;
  using Ging = NewDistGraphGeneric<NodeData, EdgeData, GingerP>;
  using Fenn = NewDistGraphGeneric<NodeData, EdgeData, FennelP>;

  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut, no transpose
  if (net.Num == 1) {
    return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                             scaleFactor, false, readFromFile,
                             localGraphFileName);
  }

  switch (partitionScheme) {
  case OEC:
    return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                             scaleFactor, false, readFromFile,
                             localGraphFileName);
  case IEC:
    if (inputFileTranspose.size()) {
      return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num,
                               scaleFactor, true, readFromFile,
                               localGraphFileName);
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
                                 net.Num, scaleFactor, true, VCutThreshold,
                                 false, readFromFile, localGraphFileName);
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
                                  scaleFactor, false, readFromFile,
                                  localGraphFileName);
  case CART_VCUT_IEC:
    return new Graph_cartesianCut(inputFileTranspose, partFolder, net.ID, net.Num,
                                  scaleFactor, true, readFromFile,
                                  localGraphFileName);
  case CART_VCUT_OLD:
    return new Graph_cartesianCutOld(inputFile, partFolder, net.ID, net.Num,
                                     scaleFactor, false, readFromFile,
                                     localGraphFileName);
  case JAGGED_CYCLIC_VCUT:
    return new Graph_jaggedCut(inputFile, partFolder, net.ID, net.Num,
                               scaleFactor, false);
  case JAGGED_BLOCKED_VCUT:
    return new Graph_jaggedBlockedCut(inputFile, partFolder, net.ID, net.Num,
                                      scaleFactor, false);
  case OVER_DECOMPOSE_2_VCUT:
    return new Graph_cartesianCut_overDecomposeBy2(
        inputFile, partFolder, net.ID, net.Num, scaleFactor, false);
  case OVER_DECOMPOSE_4_VCUT:
    return new Graph_cartesianCut_overDecomposeBy4(
        inputFile, partFolder, net.ID, net.Num, scaleFactor, false);
  case CEC:
    return new Graph_customEdgeCut(inputFile, partFolder, net.ID, net.Num,
                                   scaleFactor, vertexIDMapFileName, false);
  case GCVC:
    return new GenericCVC(inputFile, net.ID, net.Num, false, readFromFile,
                          localGraphFileName);
  case GHIVC:
    if (inputFileTranspose.size()) {
      return new GenericHVC(inputFileTranspose, net.ID, net.Num, true);
    } else {
      GALOIS_DIE("Error: attempting generic incoming hybrid cut without "
                 "transpose graph");
      break;
    }

  case GOEC:
    return new GenericEC(inputFile, net.ID, net.Num, false);

  case GING:
    if (inputFileTranspose.size()) {
      return new Ging(inputFileTranspose, net.ID, net.Num, true);
    } else {
      GALOIS_DIE("Error: attempting Ginger without transpose graph");
      break;
    }

  case FENNEL_O:
    return new Fenn(inputFile, net.ID, net.Num, false);

  case FENNEL_I:
    if (inputFileTranspose.size()) {
      return new Fenn(inputFileTranspose, net.ID, net.Num, true);
    } else {
      GALOIS_DIE("Error: attempting Fennel incoming without transpose graph");
      break;
    }

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
template <typename NodeData, typename EdgeData, bool iterateOut = true,
          typename std::enable_if<!iterateOut>::type* = nullptr>
DistGraph<NodeData, EdgeData>*
constructGraph(std::vector<unsigned>& scaleFactor) {
  typedef DistGraphEdgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef DistGraphHybridCut<NodeData, EdgeData> Graph_vertexCut;
  typedef DistGraphCartesianCut<NodeData, EdgeData, true>
      Graph_cartesianCut; // assumes pull-style
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, false, true>
      Graph_cartesianCutOld;
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, true,
                                true>
      Graph_checkerboardCut; // assumes pull-style
  typedef DistGraphJaggedCut<NodeData, EdgeData, false,
                             true>
      Graph_jaggedCut; // assumes pull-style
  typedef DistGraphJaggedCut<NodeData, EdgeData, true,
                             true>
      Graph_jaggedBlockedCut; // assumes pull-style
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, false, true, 2>
      Graph_cartesianCut_overDecomposeBy2; // assumes pull-style
  typedef DistGraphCartesianCutOld<NodeData, EdgeData, false, true, 4>
      Graph_cartesianCut_overDecomposeBy4; // assumes pull-style
  typedef DistGraphCustomEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  using GenericCVC = DistGraphGeneric<NodeData, EdgeData, GenericCVCColumnFlip>;
  using GenericHVC = DistGraphGeneric<NodeData, EdgeData, GenericHVC>;
  using GenericEC = DistGraphGeneric<NodeData, EdgeData, NoCommunication>;
  using Ging = NewDistGraphGeneric<NodeData, EdgeData, GingerP>;
  using Fenn = NewDistGraphGeneric<NodeData, EdgeData, FennelP>;

  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut
  if (net.Num == 1) {
    if (inputFileTranspose.size()) {
      return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num,
                               scaleFactor, false, readFromFile,
                               localGraphFileName);
    } else {
      fprintf(stderr, "WARNING: Loading transpose graph through in-memory "
                      "transpose to iterate over in-edges: pass in transpose "
                      "graph with -graphTranspose to avoid unnecessary "
                      "overhead.\n");
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                               scaleFactor, true, readFromFile,
                               localGraphFileName);
    }
  }

  switch (partitionScheme) {
  case OEC:
    return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                             scaleFactor, true, readFromFile,
                             localGraphFileName);
  case IEC:
    if (inputFileTranspose.size()) {
      return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num,
                               scaleFactor, false, readFromFile,
                               localGraphFileName);
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
                                 net.Num, scaleFactor, false, VCutThreshold,
                                 false, readFromFile, localGraphFileName);
    } else {
      GALOIS_DIE("Error: (hivc) iterate over in-edges without transpose graph");
      break;
    }

  case BOARD2D_VCUT:
    if (inputFileTranspose.size()) {
      return new Graph_checkerboardCut(inputFileTranspose, partFolder, net.ID,
                                       net.Num, scaleFactor, false);
    } else {
      GALOIS_DIE("Error: (cvc) iterate over in-edges without transpose graph");
      break;
    }

  case CART_VCUT:
    return new Graph_cartesianCut(inputFile, partFolder, net.ID,
                                  net.Num, scaleFactor, true, readFromFile,
                                  localGraphFileName);
  case CART_VCUT_IEC:
    if (inputFileTranspose.size()) {
      return new Graph_cartesianCut(inputFileTranspose, partFolder, net.ID,
                                    net.Num, scaleFactor, false, readFromFile,
                                    localGraphFileName);
    } else {
      GALOIS_DIE("Error: (cvc) iterate over in-edges without transpose graph");
      break;
    }
  case CART_VCUT_OLD:
    return new Graph_cartesianCutOld(inputFile, partFolder, net.ID,
                                     net.Num, scaleFactor, true, readFromFile,
                                     localGraphFileName);
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
      return new Graph_jaggedBlockedCut(inputFileTranspose, partFolder, net.ID,
                                        net.Num, scaleFactor, false);
    } else {
      GALOIS_DIE("Error: (jbvc) iterate over in-edges without transpose graph");
      break;
    }
  case OVER_DECOMPOSE_2_VCUT:
    if (inputFileTranspose.size()) {
      return new Graph_cartesianCut_overDecomposeBy2(
          inputFileTranspose, partFolder, net.ID, net.Num, scaleFactor, false);
    } else {
      GALOIS_DIE(
          "Error: (od2vc) iterate over in-edges without transpose graph");
      break;
    }
  case OVER_DECOMPOSE_4_VCUT:
    if (inputFileTranspose.size()) {
      return new Graph_cartesianCut_overDecomposeBy4(
          inputFileTranspose, partFolder, net.ID, net.Num, scaleFactor, false);
    } else {
      GALOIS_DIE(
          "Error: (od4vc) iterate over in-edges without transpose graph");
      break;
    }

  case CEC:
    if (inputFileTranspose.size()) {
      return new Graph_customEdgeCut(inputFileTranspose, partFolder, net.ID,
                                     net.Num, scaleFactor, vertexIDMapFileName,
                                     false);
    } else {
      GALOIS_DIE(
          "Error: (cec) iterate over in-edges without transpose graph");
      break;
    }

  case GCVC:
    // read regular partition and then flip it
    return new GenericCVC(inputFile, net.ID, net.Num, true, readFromFile,
                          localGraphFileName);

  case GHIVC:
    if (inputFileTranspose.size()) {
      return new GenericHVC(inputFileTranspose, net.ID, net.Num, false);
    } else {
      GALOIS_DIE("Error: attempting generic incoming hybrid cut without "
                 "transpose graph");
      break;
    }

  case GOEC:
    return new GenericEC(inputFile, net.ID, net.Num, true);

  case GING:
    if (inputFileTranspose.size()) {
      return new Ging(inputFileTranspose, net.ID, net.Num, false);
    } else {
      GALOIS_DIE("Error: attempting Ginger without transpose graph");
      break;
    }

  case FENNEL_O:
    return new Fenn(inputFile, net.ID, net.Num, true);

  case FENNEL_I:
    if (inputFileTranspose.size()) {
      return new Fenn(inputFileTranspose, net.ID, net.Num, false);
    } else {
      GALOIS_DIE("Error: attempting Fennel incoming without transpose graph");
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
