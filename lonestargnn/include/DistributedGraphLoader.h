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
 * Version for GNNs which only support symmetric graphs at this point in time.
 *
 * @todo Refactoring a bunch of this code is likely very possible to do
 */
#ifndef D_GRAPH_LOADER_SYM
#define D_GRAPH_LOADER_SYM

#include "galois/graphs/CuSPPartitioner.h"
#include "deepgalois/utils.h"

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
extern cll::opt<std::string> dataset;
//! partitioning scheme to use
extern cll::opt<galois::graphs::PARTITIONING_SCHEME> partitionScheme;

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
  std::string inputFile = deepgalois::path + dataset + ".csgr";
  galois::gInfo("File to read is ", inputFile);

  switch (partitionScheme) {
  case OEC:
  case IEC:
    return cuspPartitionGraph<NoCommunication, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, ""
    );
  case HOVC:
  case HIVC:
    return cuspPartitionGraph<GenericHVC, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, ""
    );

  case CART_VCUT:
  case CART_VCUT_IEC:
    return cuspPartitionGraph<GenericCVC, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, ""
    );

  case GINGER_O:
  case GINGER_I:
    return cuspPartitionGraph<GingerP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, ""
    );

  case FENNEL_O:
  case FENNEL_I:
    return cuspPartitionGraph<FennelP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, ""
    );

  case SUGAR_O:
    return cuspPartitionGraph<SugarP, NodeData, EdgeData>(
      inputFile, galois::CUSP_CSR, galois::CUSP_CSR, true, ""
    );
  default:
    GALOIS_DIE("Error: partition scheme specified is invalid");
    return nullptr;
  }
}

} // end namespace graphs
} // end namespace galois
#endif
