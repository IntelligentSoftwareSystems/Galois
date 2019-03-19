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

#include "galois/graphs/Generic.h"
#include "galois/graphs/NewGeneric.h"
#include "galois/graphs/DistributedGraph_CustomEdgeCut.h"
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
  CART_VCUT,             //!< cartesian vertex cut
  CART_VCUT_IEC,         //!< cartesian vertex cut using iec
  CEC,                    //!< custom edge cut
  GINGER_O,                    //!< Ginger, outgoing
  GINGER_I,                    //!< Ginger, incoming
  FENNEL_O,                   //!< Fennel, oec
  FENNEL_I,                    //!< Fennel, iec
  SUGAR_O                    //!< Sugar, oec
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
  case CART_VCUT:
    return "cvc";
  case CART_VCUT_IEC:
    return "cvc_iec";
  case CEC:
    return "cec";
  case GINGER_O:
    return "ginger-oec";
  case GINGER_I:
    return "ginger-iec";
  case FENNEL_O:
    return "fennel-oec";
  case FENNEL_I:
    return "fennel-iec";
  case SUGAR_O:
    return "sugar-oec";
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
//! partitioning scheme to use
extern cll::opt<galois::graphs::PARTITIONING_SCHEME> partitionScheme;
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

  typedef DistGraphCustomEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  using GenericEC = DistGraphGeneric<NodeData, EdgeData, NoCommunication>;
  using GenericHVC = DistGraphGeneric<NodeData, EdgeData, GenericHVC>;
  using GenericCVC = DistGraphGeneric<NodeData, EdgeData, GenericCVC>;
  using Ginger = NewDistGraphGeneric<NodeData, EdgeData, GingerP>;
  using Fennel = NewDistGraphGeneric<NodeData, EdgeData, FennelP>;
  using Sugar = NewDistGraphGeneric<NodeData, EdgeData, SugarP>;

  auto& net = galois::runtime::getSystemNetworkInterface();

  switch (partitionScheme) {
  case OEC:
  case IEC:
    return new GenericEC(inputFile, net.ID, net.Num, false, readFromFile,
                         localGraphFileName);

  case HOVC:
  case HIVC:
    return new GenericHVC(inputFile, net.ID, net.Num, false, readFromFile,
                          localGraphFileName);

  case CART_VCUT:
  case CART_VCUT_IEC:
    return new GenericCVC(inputFile, net.ID, net.Num, false,
                          readFromFile, localGraphFileName);

  case CEC:
    return new Graph_customEdgeCut(inputFile, "", net.ID, net.Num,
                                   scaleFactor, vertexIDMapFileName, false);

  case GINGER_O:
  case GINGER_I:
    return new Ginger(inputFile, net.ID, net.Num, false);

  case FENNEL_O:
  case FENNEL_I:
    return new Fennel(inputFile, net.ID, net.Num, false);

  case SUGAR_O:
    return new Sugar(inputFile, net.ID, net.Num, false);

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
  typedef DistGraphCustomEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  using GenericEC = DistGraphGeneric<NodeData, EdgeData, NoCommunication>;
  using GenericCVC = DistGraphGeneric<NodeData, EdgeData, GenericCVC>;
  using GenericHVC = DistGraphGeneric<NodeData, EdgeData, GenericHVC>;
  using Ginger = NewDistGraphGeneric<NodeData, EdgeData, GingerP>;
  using Fennel = NewDistGraphGeneric<NodeData, EdgeData, FennelP>;
  using Sugar = NewDistGraphGeneric<NodeData, EdgeData, SugarP>;

  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut, no transpose
  if (net.Num == 1) {
    return new GenericEC(inputFile, net.ID, net.Num, false, readFromFile,
                         localGraphFileName);
  }

  switch (partitionScheme) {
  case OEC:
    return new GenericEC(inputFile, net.ID, net.Num, false, readFromFile,
                         localGraphFileName);
  case IEC:
    if (inputFileTranspose.size()) {
      return new GenericEC(inputFileTranspose, net.ID, net.Num, true,
                           readFromFile, localGraphFileName);
    } else {
      GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                 "graph");
      break;
    }

  case HOVC:
    return new GenericHVC(inputFile, net.ID, net.Num, false, readFromFile,
                          localGraphFileName);
  case HIVC:
    if (inputFileTranspose.size()) {
      return new GenericHVC(inputFileTranspose, net.ID, net.Num, true,
                            readFromFile, localGraphFileName);
    } else {
      GALOIS_DIE("Error: attempting incoming hybrid cut without transpose "
                 "graph");
      break;
    }

  case CART_VCUT:
    return new GenericCVC(inputFile, net.ID, net.Num, false, readFromFile,
                          localGraphFileName);

  case CART_VCUT_IEC:
    if (inputFileTranspose.size()) {
      return new GenericCVC(inputFileTranspose, net.ID, net.Num, true, readFromFile,
                            localGraphFileName);
    } else {
      GALOIS_DIE("Error: attempting cvc incoming cut without "
                 "transpose graph");
      break;
    }

  case CEC:
    return new Graph_customEdgeCut(inputFile, "", net.ID, net.Num,
                                   scaleFactor, vertexIDMapFileName, false);

  case GINGER_O:
    return new Ginger(inputFile, net.ID, net.Num, false);

  case GINGER_I:
    if (inputFileTranspose.size()) {
      return new Ginger(inputFileTranspose, net.ID, net.Num, true);
    } else {
      GALOIS_DIE("Error: attempting Ginger without transpose graph");
      break;
    }

  case FENNEL_O:
    return new Fennel(inputFile, net.ID, net.Num, false);

  case FENNEL_I:
    if (inputFileTranspose.size()) {
      return new Fennel(inputFileTranspose, net.ID, net.Num, true);
    } else {
      GALOIS_DIE("Error: attempting Fennel incoming without transpose graph");
      break;
    }

  case SUGAR_O:
    return new Sugar(inputFile, net.ID, net.Num, false);

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
 * false, will iterate over in edges
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
  typedef DistGraphCustomEdgeCut<NodeData, EdgeData> Graph_customEdgeCut;
  using GenericEC = DistGraphGeneric<NodeData, EdgeData, NoCommunication>;
  using GenericCVC = DistGraphGeneric<NodeData, EdgeData, GenericCVCColumnFlip>;
  using GenericHVC = DistGraphGeneric<NodeData, EdgeData, GenericHVC>;
  using Ginger = NewDistGraphGeneric<NodeData, EdgeData, GingerP>;
  using Fennel = NewDistGraphGeneric<NodeData, EdgeData, FennelP>;
  using Sugar = NewDistGraphGeneric<NodeData, EdgeData, SugarColumnFlipP>;

  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut
  if (net.Num == 1) {
    if (inputFileTranspose.size()) {
      return new GenericEC(inputFileTranspose, net.ID, net.Num, false,
                           readFromFile, localGraphFileName);
    } else {
      fprintf(stderr, "WARNING: Loading transpose graph through in-memory "
                      "transpose to iterate over in-edges: pass in transpose "
                      "graph with -graphTranspose to avoid unnecessary "
                      "overhead.\n");
      return new GenericEC(inputFile, net.ID, net.Num, true, readFromFile,
                           localGraphFileName);
    }
  }

  switch (partitionScheme) {
  case OEC:
    return new GenericEC(inputFile, net.ID, net.Num, true, readFromFile,
                         localGraphFileName);
  case IEC:
    if (inputFileTranspose.size()) {
      return new GenericEC(inputFileTranspose, net.ID, net.Num, false,
                           readFromFile, localGraphFileName);
    } else {
      GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                 "graph");
      break;
    }

  case HOVC:
    return new GenericHVC(inputFile, net.ID, net.Num, true, readFromFile,
                          localGraphFileName);
  case HIVC:
    if (inputFileTranspose.size()) {
      return new GenericHVC(inputFileTranspose, net.ID, net.Num, false,
                            readFromFile, localGraphFileName);
    } else {
      GALOIS_DIE("Error: (hivc) iterate over in-edges without transpose graph");
      break;
    }

  case CART_VCUT:
    // read regular partition and then flip it
    return new GenericCVC(inputFile, net.ID, net.Num, true, readFromFile,
                          localGraphFileName);
  case CART_VCUT_IEC:
    if (inputFileTranspose.size()) {
      return new GenericCVC(inputFileTranspose, net.ID, net.Num, false,
                            readFromFile, localGraphFileName);
    } else {
      GALOIS_DIE("Error: (cvc) iterate over in-edges without transpose graph");
      break;
    }

  case CEC:
    if (inputFileTranspose.size()) {
      return new Graph_customEdgeCut(inputFileTranspose, "", net.ID,
                                     net.Num, scaleFactor, vertexIDMapFileName,
                                     false);
    } else {
      GALOIS_DIE("Error: (cec) iterate over in-edges without transpose graph");
      break;
    }


  case GINGER_O:
    return new Ginger(inputFile, net.ID, net.Num, true);
  case GINGER_I:
    if (inputFileTranspose.size()) {
      return new Ginger(inputFileTranspose, net.ID, net.Num, false);
    } else {
      GALOIS_DIE("Error: attempting Ginger without transpose graph");
      break;
    }

  case FENNEL_O:
    return new Fennel(inputFile, net.ID, net.Num, true);
  case FENNEL_I:
    if (inputFileTranspose.size()) {
      return new Fennel(inputFileTranspose, net.ID, net.Num, false);
    } else {
      GALOIS_DIE("Error: attempting Fennel incoming without transpose graph");
      break;
    }

  case SUGAR_O:
    return new Sugar(inputFile, net.ID, net.Num, true);

  default:
    GALOIS_DIE("Error: partition scheme specified is invalid");
    return nullptr;
  }
}

} // end namespace graphs
} // end namespace galois
#endif
