/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
 * @file Reader.h
 *
 * Contains definitions for the common distributed graph loading functionality
 * of benchmark applications.
 */
#ifndef GALOIS_DISTBENCH_INPUT_H
#define GALOIS_DISTBENCH_INPUT_H

#include "galois/graphs/CuSPPartitioner.h"
#include "llvm/Support/CommandLine.h"

/*******************************************************************************
 * Supported partitioning schemes
 ******************************************************************************/

//! enums of partitioning schemes supported
enum PARTITIONING_SCHEME {
  OEC,           //!< outgoing edge cut
  IEC,           //!< incoming edge cut
  HOVC,          //!< outgoing hybrid vertex cut
  HIVC,          //!< incoming hybrid vertex cut
  CART_VCUT,     //!< cartesian vertex cut
  CART_VCUT_IEC, //!< cartesian vertex cut using iec
  // CEC,                   //!< custom edge cut
  GINGER_O, //!< Ginger, outgoing
  GINGER_I, //!< Ginger, incoming
  FENNEL_O, //!< Fennel, oec
  FENNEL_I, //!< Fennel, iec
  SUGAR_O   //!< Sugar, oec
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
  // case CEC:
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
    GALOIS_DIE("unsupported partition scheme: ", e);
  }
}

/*******************************************************************************
 * Graph-loading-related command line arguments
 ******************************************************************************/
namespace cll = llvm::cl;

//! input graph file
extern cll::opt<std::string> inputFile;
//! input graph file, but transposed
extern cll::opt<std::string> inputFileTranspose;
//! symmetric input graph file
extern cll::opt<bool> symmetricGraph;
//! partitioning scheme to use
extern cll::opt<PARTITIONING_SCHEME> partitionScheme;
//! true if input graph file format is SHAD WMD
extern cll::opt<bool> useWMD;
////! path to vertex id map for custom edge cut
// extern cll::opt<std::string> vertexIDMapFileName;
//! true if you want to read graph structure from a file
extern cll::opt<bool> readFromFile;
//! path to local graph structure to read
extern cll::opt<std::string> localGraphFileName;
//! if true, the local graph structure will be saved to disk after partitioning
extern cll::opt<bool> saveLocalGraph;
//! file specifying blocking of masters
extern cll::opt<std::string> mastersFile;

// @todo command line argument for read balancing across hosts

/*******************************************************************************
 * Graph-loading functions
 ******************************************************************************/

template <typename NodeData, typename EdgeData>
using DistGraphPtr =
    std::unique_ptr<galois::graphs::DistGraph<NodeData, EdgeData>>;

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
DistGraphPtr<NodeData, EdgeData>
constructSymmetricGraph(std::vector<unsigned>& GALOIS_UNUSED(scaleFactor)) {
  if (!symmetricGraph) {
    GALOIS_DIE("application requires a symmetric graph input;"
               " please use the -symmetricGraph flag "
               " to indicate the input is a symmetric graph");
  }

  switch (partitionScheme) {
  case OEC:
  case IEC:
    return galois::cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, true,
        inputFileTranspose, mastersFile);
  case HOVC:
  case HIVC:
    return galois::cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, true,
        inputFileTranspose);

  case CART_VCUT:
  case CART_VCUT_IEC:
    return galois::cuspPartitionGraph<GenericCVC, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, true,
        inputFileTranspose);

    // case CEC:
    //  return new Graph_customEdgeCut(inputFile, "", net.ID, net.Num,
    //                                 scaleFactor, vertexIDMapFileName, false);

  case GINGER_O:
  case GINGER_I:
    return galois::cuspPartitionGraph<GingerP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD ,true,
        inputFileTranspose);

  case FENNEL_O:
  case FENNEL_I:
    return galois::cuspPartitionGraph<FennelP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, true,
        inputFileTranspose);

  case SUGAR_O:
    return galois::cuspPartitionGraph<SugarP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, true,
        inputFileTranspose);
  default:
    GALOIS_DIE("partition scheme specified is invalid: ", partitionScheme);
    return DistGraphPtr<NodeData, EdgeData>(nullptr);
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
DistGraphPtr<NodeData, EdgeData>
constructGraph(std::vector<unsigned>& GALOIS_UNUSED(scaleFactor)) {
  // 1 host = no concept of cut; just load from edgeCut, no transpose
  auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.Num == 1) {
    return galois::cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, false,
        inputFileTranspose);
  }

  switch (partitionScheme) {
  case OEC:
    return galois::cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, false,
        inputFileTranspose, mastersFile);
  case IEC:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSR, useWMD, false,
          inputFileTranspose, mastersFile);
    } else {
      GALOIS_DIE("incoming edge cut requires transpose graph");
      break;
    }

  case HOVC:
    return galois::cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, false,
        inputFileTranspose);
  case HIVC:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSR, useWMD, false,
          inputFileTranspose);
    } else {
      GALOIS_DIE("incoming hybrid cut requires transpose graph");
      break;
    }

  case CART_VCUT:
    return galois::cuspPartitionGraph<GenericCVC, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, false,
        inputFileTranspose);

  case CART_VCUT_IEC:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<GenericCVC, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSR, useWMD, false,
          inputFileTranspose);
    } else {
      GALOIS_DIE("cvc incoming cut requires transpose graph");
      break;
    }

    // case CEC:
    //  return new Graph_customEdgeCut(inputFile, "", net.ID, net.Num,
    //                                 scaleFactor, vertexIDMapFileName, false);

  case GINGER_O:
    return galois::cuspPartitionGraph<GingerP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, false,
        inputFileTranspose);
  case GINGER_I:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<GingerP, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSR, useWMD, false,
          inputFileTranspose);
    } else {
      GALOIS_DIE("Ginger requires transpose graph");
      break;
    }

  case FENNEL_O:
    return galois::cuspPartitionGraph<FennelP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, false,
        inputFileTranspose);
  case FENNEL_I:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<FennelP, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSR, useWMD, false,
          inputFileTranspose);
    } else {
      GALOIS_DIE("Fennel requires transpose graph");
      break;
    }

  case SUGAR_O:
    return galois::cuspPartitionGraph<SugarP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSR, useWMD, false,
        inputFileTranspose);

  default:
    GALOIS_DIE("partition scheme specified is invalid: ", partitionScheme);
    return DistGraphPtr<NodeData, EdgeData>(nullptr);
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
DistGraphPtr<NodeData, EdgeData> constructGraph(std::vector<unsigned>&) {
  auto& net = galois::runtime::getSystemNetworkInterface();

  // 1 host = no concept of cut; just load from edgeCut
  if (net.Num == 1) {
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSC, useWMD, false,
          inputFileTranspose);
    } else {
      fprintf(stderr, "WARNING: Loading transpose graph through in-memory "
                      "transpose to iterate over in-edges: pass in transpose "
                      "graph with -graphTranspose to avoid unnecessary "
                      "overhead.\n");
      return galois::cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSR, galois::CUSP_CSC, useWMD, false,
          inputFileTranspose);
    }
  }

  switch (partitionScheme) {
  case OEC:
    return galois::cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSC, useWMD, false,
        inputFileTranspose, mastersFile);
  case IEC:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSC, useWMD, false,
          inputFileTranspose, mastersFile);
    } else {
      GALOIS_DIE("iec requires transpose graph");
      break;
    }

  case HOVC:
    return galois::cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSC, useWMD, false,
        inputFileTranspose);
  case HIVC:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSC, useWMD, false,
          inputFileTranspose);
    } else {
      GALOIS_DIE("hivc requires transpose graph");
      break;
    }

  case CART_VCUT:
    return galois::cuspPartitionGraph<GenericCVCColumnFlip, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSC, useWMD, false,
        inputFileTranspose);
  case CART_VCUT_IEC:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<GenericCVCColumnFlip, NodeData,
                                        EdgeData>(inputFile, galois::CUSP_CSC,
                                                  galois::CUSP_CSC, useWMD,
                                                  false,
                                                  inputFileTranspose);
    } else {
      GALOIS_DIE("cvc requires transpose graph");
      break;
    }

  case GINGER_O:
    return galois::cuspPartitionGraph<GingerP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSC, useWMD, false,
        inputFileTranspose);
  case GINGER_I:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<GingerP, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSC, useWMD, false,
          inputFileTranspose);
    } else {
      GALOIS_DIE("Ginger requires transpose graph");
      break;
    }

  case FENNEL_O:
    return galois::cuspPartitionGraph<FennelP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSC, useWMD, false,
        inputFileTranspose);
  case FENNEL_I:
    if (inputFileTranspose.size()) {
      return galois::cuspPartitionGraph<FennelP, NodeData, EdgeData>(
          inputFile, galois::CUSP_CSC, galois::CUSP_CSC, useWMD, false,
          inputFileTranspose);
    } else {
      GALOIS_DIE("Fennel requires transpose graph");
      break;
    }

  case SUGAR_O:
    return galois::cuspPartitionGraph<SugarColumnFlipP, NodeData, EdgeData>(
        inputFile, galois::CUSP_CSR, galois::CUSP_CSC, useWMD, false,
        inputFileTranspose);

  default:
    GALOIS_DIE("partition scheme specified is invalid: ", partitionScheme);
    return DistGraphPtr<NodeData, EdgeData>(nullptr);
  }
}

#endif
