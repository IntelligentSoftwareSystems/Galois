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
 * @file CuSPPartitioner.h
 *
 * Contains the main CuSP partitioning function.
 */

#ifndef _GALOIS_CUSP_
#define _GALOIS_CUSP_

#include "galois/DistGalois.h"
#include "galois/graphs/DistributedGraph.h"
#include "galois/graphs/NewGeneric.h"
#include "galois/graphs/GenericPartitioners.h"

namespace galois {
//! Enum for the input/output format of the partitioner.
enum CUSP_GRAPH_TYPE {
  CUSP_CSR, //!< Compressed sparse row graph format, i.e. outgoing edges
  CUSP_CSC  //!< Compressed sparse column graph format, i.e. incoming edges
};

/**
 * Main CuSP function: partitions a graph on disk, one partition per host.
 *
 * @param graphFile Graph file to read in the Galois binary CSR format
 * @param inputType Specifies which input format (CSR or CSC) should be given
 * to the partitioner
 * @param outputType Specifies the output format (CSR or CSC) that each
 * partition will be created in
 * @param symmetricGraph This should be "true" if the passed in graphFile
 * is a symmetric graph
 * @param transposeGraphFile Transpose graph of graphFile in Galois binary
 * CSC format (i.e. give it the transpose version of graphFile). Ignore
 * this argument if the graph is symmetric.
 * @param cuspAsync Toggles asynchronous master assignment phase during
 * partitioning
 * @param cuspStateRounds Toggles number of rounds used to synchronize
 * partitioning state during master assignment phase
 * @param readPolicy Determines how each host should divide the reading
 * load of the graph on disk
 * @param nodeWeight When using a read policy that involves nodes and edges,
 * this argument assigns a weight to give each node.
 * @param edgeWeight When using a read policy that involves nodes and edges,
 * this argument assigns a weight to give each edge.
 *
 * @tparam PartitionPolicy Partitioning policy object that specifies the
 * placement of nodes/edges during partitioning.
 * @tparam NodeData Data structure to be created for each node in the graph
 * @tparam EdgeData Type of data to be stored on each edge. Currently
 * only guarantee support for void or uint32_t; all other types may cause
 * undefined behavior.
 *
 * @returns A local partition of the passed in graph as a DistributedGraph
 *
 * @todo Look into making void node data work in LargeArray for D-Galois;
 * void specialization. For now, use char as default type
 */
template <typename PartitionPolicy, typename NodeData = char,
          typename EdgeData = void>
galois::graphs::DistGraph<NodeData, EdgeData>*
cuspPartitionGraph(std::string graphFile, CUSP_GRAPH_TYPE inputType,
                   CUSP_GRAPH_TYPE outputType, bool symmetricGraph = false,
                   std::string transposeGraphFile = "", bool cuspAsync = true,
                   uint32_t cuspStateRounds = 100,
                   galois::graphs::MASTERS_DISTRIBUTION readPolicy =
                       galois::graphs::BALANCED_EDGES_OF_MASTERS,
                   uint32_t nodeWeight = 0, uint32_t edgeWeight = 0) {
  auto& net = galois::runtime::getSystemNetworkInterface();
  using DistGraphConstructor =
      galois::graphs::NewDistGraphGeneric<NodeData, EdgeData, PartitionPolicy>;

  // TODO @todo bring back graph saving/reading functionality?
  std::string localGraphName = "";

  if (!symmetricGraph) {
    // out edges or in edges
    std::string inputToUse;
    // depending on output type may need to transpose edges
    bool useTranspose;

    // see what input is specified
    if (inputType == CUSP_CSR) {
      inputToUse = graphFile;
      if (outputType == CUSP_CSR) {
        useTranspose = false;
      } else if (outputType == CUSP_CSC) {
        useTranspose = true;
      } else {
        GALOIS_DIE("CuSP output graph type is invalid");
      }
    } else if (inputType == CUSP_CSC) {
      inputToUse = transposeGraphFile;
      if (outputType == CUSP_CSR) {
        useTranspose = true;
      } else if (outputType == CUSP_CSC) {
        useTranspose = false;
      } else {
        GALOIS_DIE("CuSP output graph type is invalid");
      }
    } else {
      GALOIS_DIE("Invalid input graph type specified in CuSP partitioner");
    }

    return new DistGraphConstructor(inputToUse, net.ID, net.Num, cuspAsync,
                                    cuspStateRounds, useTranspose, readPolicy,
                                    nodeWeight, edgeWeight);
  } else {
    // symmetric graph path: assume the passed in graphFile is a symmetric
    // graph; output is also symmetric
    return new DistGraphConstructor(graphFile, net.ID, net.Num, cuspAsync,
                                    cuspStateRounds, false, readPolicy,
                                    nodeWeight, edgeWeight);
  }
}
} // end namespace galois
#endif
