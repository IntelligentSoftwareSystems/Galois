/** Graph helper functions -*- C++ -*-
 * @file GraphHelpers.h
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Contains functions that can be done on various graphs with a particular
 * interface as well as functions that operate nodes/edges of a graph.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef __GALOIS_GRAPH_HELPERS__
#define __GALOIS_GRAPH_HELPERS__

#include <galois/gIO.h>

#include <boost/iterator/counting_iterator.hpp>
#include <cassert>
#include <vector>

namespace galois {
namespace graphs {

namespace internal {
/**
 * Return a suitable index between an upper bound and a lower bound that
 * attempts to get close to the target size (i.e. find a good chunk that
 * corresponds to some size) using a prefix sum.
 *
 * @tparam PrefixSumType type of the object that holds the edge prefix sum
 *
 * @param nodeWeight weight to give to a node in division
 * @param edgeWeight weight to give to an edge in division
 * @param targetWeight The amount of weight we want from the returned index
 * @param lb lower bound to start search from
 * @param ub upper bound to start search from
 * @param edgePrefixSum prefix sum of edges; may be full or partial prefix
 * sum of the object you are attempting to split
 * @param edgeOffset number of edges to subtract from edge count retrieved
 * from prefix sum; used if array is a partial prefix sum
 * @param nodeOffset number of nodes to skip over when looking in the
 * prefix sum: useful if the prefix sum is over the entire graph while you
 * just want to divide the nodes for a particular region (jump to the region
 * with the nodeOffset)
 *
 * @returns The node id that hits (or gets close to) the target size
 */
// Note: "inline" may be required if PrefixSumType is exactly the same type
// in 2 different translation units; otherwise it should be fine
template <typename PrefixSumType>
size_t findIndexPrefixSum(size_t nodeWeight, size_t edgeWeight,
                          size_t targetWeight, uint64_t lb, uint64_t ub,
                          PrefixSumType& edgePrefixSum,
                          uint64_t edgeOffset, uint64_t nodeOffset) {
  assert(nodeWeight != 0 || edgeWeight != 0);

  while (lb < ub) {
    size_t mid = lb + (ub - lb) / 2;
    size_t num_edges;

    if ((mid + nodeOffset) != 0) {
      num_edges = edgePrefixSum[mid - 1 + nodeOffset] - edgeOffset;
    } else {
      num_edges = 0;
    }

    size_t weight = num_edges * edgeWeight + mid * nodeWeight;

    if (weight <= targetWeight) {
      lb = mid + 1;
    } else {
      ub = mid;
    }
  }

  return lb;
}

/**
 * Given a number of divisions and a scale factor specifying how much of a
 * chunk of blocks each division should get, determine the total number
 * of blocks to split among all divisions + calculate the prefix sum and
 * save it in-place to the scaleFactor variable.
 *
 * @param numDivisions number of divisions to split blocks among
 * @param scaleFactor vector specifying how much a particular vision should get
 *
 * @returns The total number of blocks to split among all divisions
 */
// inline required as GraphHelpers is included in multiple translation units
uint32_t determine_block_division(uint32_t numDivisions,
                                  std::vector<unsigned>& scaleFactor);



} // end namespace internal

/**
 * Returns 2 ranges (one for nodes, one for edges) for a particular division.
 * The ranges specify the nodes/edges that a division is responsible for. The
 * function attempts to split them evenly among threads given some kind of
 * weighting for both nodes and edges.
 *
 * Assumes the parameters passed in apply to a local portion of whatever
 * is being divided (i.e. concept of a "global" object is abstracted away in
 * some sense)
 *
 * @tparam PrefixSumType type of the object that holds the edge prefix sum
 * @tparam NodeType size of the type representing the node
 *
 * @param nodeWeight weight to give to a node in division
 * @param edgeWeight weight to give to an edge in division
 * @param id Division number you want the range for
 * @param total Total number of divisions to divide nodes among
 * @param edgePrefixSum Prefix sum of the edges in the graph
 * @param scaleFactor Vector specifying if certain divisions should get more
 * than other divisions
 * @param edgeOffset number of edges to subtract from numbers in edgePrefixSum
 * @param nodeOffset number of nodes to skip over when looking in the
 * prefix sum: useful if the prefix sum is over the entire graph while you
 * just want to divide the nodes for a particular region (jump to the region
 * with the nodeOffset)
 *
 * @returns A node pair and an edge pair specifying the assigned nodes/edges
 * to division "id"; returns LOCAL ids, not global ids (i.e. if node offset
 * was used, it is up to the caller to add the offset to the numbers)
 */
// Note: "inline" may be required if PrefixSumType is exactly the same type
// in 2 different translation units; otherwise it should be fine
// If inline is used, then apparently you cannot use typedefs, so get rid
// of those if the need arises.
template <typename PrefixSumType, typename NodeType = uint64_t>
auto divideNodesBinarySearch(NodeType numNodes, uint64_t numEdges,
                   size_t nodeWeight, size_t edgeWeight, size_t id,
                   size_t total, PrefixSumType& edgePrefixSum,
                   std::vector<unsigned> scaleFactor = std::vector<unsigned>(),
                   uint64_t edgeOffset = 0, uint64_t nodeOffset = 0) {
  typedef boost::counting_iterator<NodeType> iterator;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef std::pair<iterator, iterator> NodeRange;
  typedef std::pair<edge_iterator, edge_iterator> EdgeRange;
  typedef std::pair<NodeRange, EdgeRange> GraphRange;

  // numNodes = 0 corner case
  if (numNodes == 0) {
    return GraphRange(NodeRange(iterator(0),
                                iterator(0)),
                      EdgeRange(edge_iterator(0),
                                edge_iterator(0)));
  }

  assert(nodeWeight != 0 || edgeWeight != 0);
  assert(total >= 1);
  assert(id >= 0 && id < total);

  // weight of all data
  uint64_t weight = numNodes * nodeWeight + numEdges * edgeWeight;
  // determine the number of blocks to divide among total divisions + setup the
  // scale factor vector if necessary
  uint32_t numBlocks = internal::determine_block_division(total, scaleFactor);
  // weight of a block (one block for each division by default; if scale
  // factor specifies something different, then use that instead)
  uint64_t blockWeight = (weight + numBlocks - 1) / numBlocks;

  // lower and upper blocks that this division should use determined
  // using scaleFactor
  uint32_t blockLower;
  if (id != 0) {
    blockLower = scaleFactor[id - 1];
  } else {
    blockLower = 0;
  }

  uint32_t blockUpper = scaleFactor[id];

  assert(blockLower <= blockUpper);

  uint64_t nodesLower;
  // use prefix sum to find node bounds
  if (blockLower == 0) {
    nodesLower = 0;
  } else {
    nodesLower = internal::findIndexPrefixSum(nodeWeight, edgeWeight,
                                    blockWeight * blockLower, 0, numNodes,
                                    edgePrefixSum, edgeOffset, nodeOffset);
  }

  uint64_t nodesUpper;
  nodesUpper = internal::findIndexPrefixSum(nodeWeight, edgeWeight,
                                  blockWeight * blockUpper, nodesLower,
                                  numNodes, edgePrefixSum, edgeOffset,
                                  nodeOffset);

  // get the edges bounds using node lower/upper bounds
  uint64_t edgesLower = numEdges;
  uint64_t edgesUpper = numEdges;

  if (nodesLower != nodesUpper) {
    if ((nodesLower + nodeOffset) != 0) {
      edgesLower = edgePrefixSum[nodesLower - 1 + nodeOffset] - edgeOffset;
    } else {
      edgesLower = 0;
    }

    edgesUpper = edgePrefixSum[nodesUpper - 1 + nodeOffset] - edgeOffset;
  }

  return GraphRange(NodeRange(iterator(nodesLower),
                              iterator(nodesUpper)),
                    EdgeRange(edge_iterator(edgesLower),
                              edge_iterator(edgesUpper)));
}

// second internal namespace
namespace internal {
/**
 * Helper function used by determineGraphRanges that consists of the main
 * loop over all units and calls to divide by node to determine the
 * division of nodes to units.
 *
 * Saves the ranges to an argument vector provided by the caller.
 *
 * @param beginNode Beginning of range
 * @param endNode End of range, non-inclusive
 * @param returnRanges Vector to store thread offsets for ranges in
 * @param nodeAlpha The higher the number, the more weight nodes have in
 * determining division of nodes (edges have weight 1).
 */
template <typename GraphTy>
void determineUnitRangesLoop(GraphTy& graph, uint32_t unitsToSplit,
                             uint32_t beginNode, uint32_t endNode,
                             std::vector<uint32_t>& returnRanges,
                             uint32_t nodeAlpha) {
  uint32_t numNodesInRange = endNode - beginNode;
  uint64_t numEdgesInRange = graph.edge_end(endNode - 1) - 
                             graph.edge_begin(beginNode);
  uint64_t edgeOffset = *graph.edge_begin(beginNode);

  returnRanges[0] = beginNode;
  std::vector<unsigned int> dummyScaleFactor;

  for (uint32_t i = 0; i < unitsToSplit; i++) {
    // determine division for unit i
    auto nodeSplits = 
      divideNodesBinarySearch<GraphTy, uint32_t>(
        numNodesInRange, numEdgesInRange, nodeAlpha, 1, i, unitsToSplit, 
        graph, dummyScaleFactor, edgeOffset, beginNode
      ).first;

    // i.e. if there are actually assigned nodes
    if (nodeSplits.first != nodeSplits.second) {
      if (i != 0) {
        assert(returnRanges[i] == *(nodeSplits.first));
      } else { // i == 0
        assert(returnRanges[i] == beginNode);
      }
      returnRanges[i + 1] = *(nodeSplits.second) + beginNode;
    } else {
      // thread assinged no nodes
      returnRanges[i + 1] = returnRanges[i];
    }

    galois::gDebug("Unit ", i, " gets nodes ", returnRanges[i], " to ", 
                   returnRanges[i + 1], ", num edges is ", 
                   graph.edge_end(returnRanges[i + 1] - 1) -
                   graph.edge_begin(returnRanges[i]));
  }
}
} // end second internal namespace

/**
 * Determines node division ranges for a given range of nodes and returns it 
 * as an offset vector. (node ranges = assigned nodes that a particular unit
 * of execution should work on)
 *
 * Checks for corner cases, then calls the main loop function.
 *
 * ONLY CALL AFTER GRAPH IS CONSTRUCTED as it uses functions that assume
 * the graph is already constructed.
 *
 * @param beginNode Beginning of range
 * @param endNode End of range, non-inclusive
 * @param returnRanges Vector to store thread offsets for ranges in
 * @param nodeAlpha The higher the number, the more weight nodes have in
 * determining division of nodes (edges have weight 1).
 */
template<typename GraphTy>
std::vector<uint32_t> determineUnitRangesFromGraph(GraphTy& graph,
                                                   uint32_t unitsToSplit,
                                                   uint32_t beginNode,
                                                   uint32_t endNode,
                                                   uint32_t nodeAlpha=0) {
  uint32_t total_nodes = endNode - beginNode;

  std::vector<uint32_t> returnRanges;
  returnRanges.resize(unitsToSplit + 1);

  // check corner cases
  // no nodes = assign nothing to all units
  if (beginNode == endNode) {
    returnRanges[0] = beginNode;

    for (uint32_t i = 0; i < unitsToSplit; i++) {
      returnRanges[i+1] = beginNode;
    }

    return returnRanges;
  }

  // single thread case; 1 unit gets all
  if (unitsToSplit == 1) {
    returnRanges[0] = beginNode;
    returnRanges[1] = endNode;
    return returnRanges;
  // more units than nodes
  } else if (unitsToSplit > total_nodes) {
    uint32_t current_node = beginNode;
    returnRanges[0] = current_node;
    // 1 node for units until out of threads
    for (uint32_t i = 0; i < total_nodes; i++) {
      returnRanges[i + 1] = ++current_node;
    }
    // deal with remainder units; they get nothing
    for (uint32_t i = total_nodes; i < unitsToSplit; i++) {
      returnRanges[i + 1] = total_nodes;
    }

    return returnRanges;
  }

  // no corner cases: onto main loop over nodes that determines
  // node ranges
  internal::determineUnitRangesLoop(graph, unitsToSplit, beginNode, endNode, 
                                    returnRanges, nodeAlpha);

  #ifndef NDEBUG
  // sanity checks
  assert(returnRanges[0] == beginNode &&
         "return ranges begin not the begin node");
  assert(returnRanges[unitsToSplit] == endNode &&
         "return ranges end not end node");

  for (uint32_t i = 1; i < unitsToSplit; i++) {
    assert(returnRanges[i] >= beginNode && returnRanges[i] <= endNode);
    assert(returnRanges[i] >= returnRanges[i - 1]);
  }
  #endif

  return returnRanges;
}

/**
 * Uses the divideByNode function (which is binary search based) to
 * divide nodes among units.
 *
 * @param unitsToSplit number of units to split nodes among
 * @param edgePrefixSum A prefix sum of edges
 * @returns vector that indirectly specifies how nodes are split amongs units
 * of execution
 */
template <typename VectorTy>
std::vector<uint32_t> determineUnitRangesFromPrefixSum(uint32_t unitsToSplit,  
                                                      VectorTy& edgePrefixSum) {
  assert(unitsToSplit > 0);

  std::vector<uint32_t> nodeRanges;
  nodeRanges.resize(unitsToSplit + 1);

  nodeRanges[0] = 0;

  uint32_t numNodes = edgePrefixSum.size();
  uint32_t numEdges = edgePrefixSum[numNodes - 1];

  for (uint32_t i = 0; i < unitsToSplit; i++) {
    auto nodeSplits = 
      divideNodesBinarySearch<VectorTy, uint32_t>(
        numNodes, numEdges, 0, 1, i, unitsToSplit, edgePrefixSum
      ).first;

    // i.e. if there are actually assigned nodes
    if (nodeSplits.first != nodeSplits.second) {
      if (i != 0) {
        assert(nodeRanges[i] == *(nodeSplits.first));
      } else { // i == 0
        assert(nodeRanges[i] == 0);
      }
      nodeRanges[i + 1] = *(nodeSplits.second);
    } else {
      // thread assinged no nodes
      nodeRanges[i + 1] = nodeRanges[i];
    }

    galois::gDebug("Unit ", i, " gets nodes ", nodeRanges[i], " to ",
                   nodeRanges[i+1]);
  }

  return nodeRanges;
}

} // end namespace graphs
} // end namespace galois
#endif
