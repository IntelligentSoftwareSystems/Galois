/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2019, The University of Texas at Austin. All rights reserved.
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

#include "galois/graphs/CuSPPartitioner.h"

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
  //CEC,                   //!< custom edge cut
  GINGER_O,              //!< Ginger, outgoing
  GINGER_I,              //!< Ginger, incoming
  FENNEL_O,              //!< Fennel, oec
  FENNEL_I,              //!< Fennel, iec
  SUGAR_O                //!< Sugar, oec
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
  //case CEC:
  //  return "cec";
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
////! path to vertex id map for custom edge cut
//extern cll::opt<std::string> vertexIDMapFileName;
//! true if you want to read graph structure from a file
extern cll::opt<bool> readFromFile;
//! path to local graph structure to read
extern cll::opt<std::string> localGraphFileName;
//! if true, the local graph structure will be saved to disk after partitioning
extern cll::opt<bool> saveLocalGraph;
//! file specifying blocking of masters
extern cll::opt<std::string> mastersFile;

// @todo command line argument for read balancing across hosts

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

  switch (partitionScheme) {
  case OEC:
  case IEC:
    return cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, inputFileTranspose,
      mastersFile
    );
  case HOVC:
  case HIVC:
    return cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, inputFileTranspose
    );

  case CART_VCUT:
  case CART_VCUT_IEC:
    return cuspPartitionGraph<GenericCVC, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, inputFileTranspose
    );

  //case CEC:
  //  return new Graph_customEdgeCut(inputFile, "", net.ID, net.Num,
  //                                 scaleFactor, vertexIDMapFileName, false);

  case GINGER_O:
  case GINGER_I:
    return cuspPartitionGraph<GingerP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, inputFileTranspose
    );

  case FENNEL_O:
  case FENNEL_I:
    return cuspPartitionGraph<FennelP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, inputFileTranspose
    );

  case SUGAR_O:
    return cuspPartitionGraph<SugarP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, inputFileTranspose
    );
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
  // 1 host = no concept of cut; just load from edgeCut, no transpose
  auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.Num == 1) {
    return cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, false, inputFileTranspose
    );
  }

  switch (partitionScheme) {
  case OEC:
    return cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, false, inputFileTranspose,
      mastersFile
    );
  case IEC:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSR, false, inputFileTranspose,
        mastersFile
      );
    } else {
      GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                 "graph");
      break;
    }

  case HOVC:
    return cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, false, inputFileTranspose
    );
  case HIVC:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSR, false, inputFileTranspose
      );
    } else {
      GALOIS_DIE("Error: attempting incoming hybrid cut without transpose "
                 "graph");
      break;
    }

  case CART_VCUT:
    return cuspPartitionGraph<GenericCVC, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, false, inputFileTranspose
    );

  case CART_VCUT_IEC:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<GenericCVC, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSR, false, inputFileTranspose
      );
    } else {
      GALOIS_DIE("Error: attempting cvc incoming cut without "
                 "transpose graph");
      break;
    }

  //case CEC:
  //  return new Graph_customEdgeCut(inputFile, "", net.ID, net.Num,
  //                                 scaleFactor, vertexIDMapFileName, false);

  case GINGER_O:
    return cuspPartitionGraph<GingerP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, false, inputFileTranspose
    );
  case GINGER_I:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<GingerP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSR, false, inputFileTranspose
      );
    } else {
      GALOIS_DIE("Error: attempting Ginger without transpose graph");
      break;
    }

  case FENNEL_O:
    return cuspPartitionGraph<FennelP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, false, inputFileTranspose
    );
  case FENNEL_I:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<FennelP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSR, false, inputFileTranspose
      );
    } else {
      GALOIS_DIE("Error: attempting Fennel incoming without transpose graph");
      break;
    }

  case SUGAR_O:
    return cuspPartitionGraph<SugarP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, false, inputFileTranspose
    );

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
  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut
  if (net.Num == 1) {
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSC, false, inputFileTranspose
      );
    } else {
      fprintf(stderr, "WARNING: Loading transpose graph through in-memory "
                      "transpose to iterate over in-edges: pass in transpose "
                      "graph with -graphTranspose to avoid unnecessary "
                      "overhead.\n");
      return cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSC, false, inputFileTranspose
      );
    }
  }

  switch (partitionScheme) {
  case OEC:
    return cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSC, false, inputFileTranspose,
      mastersFile
    );
  case IEC:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSC, false, inputFileTranspose,
        mastersFile
      );
    } else {
      GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                 "graph");
      break;
    }

  case HOVC:
    return cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSC, false, inputFileTranspose
    );
  case HIVC:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSC, false, inputFileTranspose
      );
    } else {
      GALOIS_DIE("Error: (hivc) iterate over in-edges without transpose graph");
      break;
    }

  case CART_VCUT:
    return cuspPartitionGraph<GenericCVCColumnFlip, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSC, false, inputFileTranspose
    );
  case CART_VCUT_IEC:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<GenericCVCColumnFlip, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSC, false, inputFileTranspose
      );
    } else {
      GALOIS_DIE("Error: (cvc) iterate over in-edges without transpose graph");
      break;
    }

  //case CEC:
  //  if (inputFileTranspose.size()) {
  //    return new Graph_customEdgeCut(inputFileTranspose, "", net.ID,
  //                                   net.Num, scaleFactor, vertexIDMapFileName,
  //                                   false);
  //  } else {
  //    GALOIS_DIE("Error: (cec) iterate over in-edges without transpose graph");
  //    break;
  //  }

  case GINGER_O:
    return cuspPartitionGraph<GingerP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSC, false, inputFileTranspose
    );
  case GINGER_I:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<GingerP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSC, false, inputFileTranspose
      );
    } else {
      GALOIS_DIE("Error: attempting Ginger without transpose graph");
      break;
    }

  case FENNEL_O:
    return cuspPartitionGraph<FennelP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSC, false, inputFileTranspose
    );
  case FENNEL_I:
    if (inputFileTranspose.size()) {
      return cuspPartitionGraph<FennelP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSC, galois::CUSP_CSC, false, inputFileTranspose
      );
    } else {
      GALOIS_DIE("Error: attempting Fennel incoming without transpose graph");
      break;
    }

  case SUGAR_O:
    return cuspPartitionGraph<SugarColumnFlipP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSC, false, inputFileTranspose
    );

  default:
    GALOIS_DIE("Error: partition scheme specified is invalid");
    return nullptr;
  }
}

} // end namespace graphs
} // end namespace galois
#endif
