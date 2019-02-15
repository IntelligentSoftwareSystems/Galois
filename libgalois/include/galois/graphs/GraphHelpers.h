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
                          PrefixSumType& edgePrefixSum, uint64_t edgeOffset,
                          uint64_t nodeOffset) {
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

    if (weight < targetWeight) {
      lb = mid + 1;
    } else if (weight == targetWeight) {
      return mid;
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
uint32_t determine_block_division(uint32_t numDivisions,
                                  std::vector<unsigned>& scaleFactor);

} // end namespace internal

/**
 * Returns 2 ranges (one for nodes, one for edges) for a particular division.
 * The ranges specify the nodes/edges that a division is responsible for. The
 * function attempts to split them evenly among units given some kind of
 * weighting for both nodes and edges.
 *
 * Assumes the parameters passed in apply to a local portion of whatever
 * is being divided (i.e. concept of a "global" object is abstracted away in
 * some sense)
 *
 * @tparam PrefixSumType type of the object that holds the edge prefix sum
 * @tparam NodeType size of the type representing the node
 *
 * @param numNodes Total number of nodes included in prefix sum
 * @param numEdges Total number of edges included in prefix sum
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
auto divideNodesBinarySearch(
    NodeType numNodes, uint64_t numEdges, size_t nodeWeight, size_t edgeWeight,
    size_t id, size_t total, PrefixSumType& edgePrefixSum,
    std::vector<unsigned> scaleFactor = std::vector<unsigned>(),
    uint64_t edgeOffset = 0, uint64_t nodeOffset = 0) {
  typedef boost::counting_iterator<NodeType> iterator;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef std::pair<iterator, iterator> NodeRange;
  typedef std::pair<edge_iterator, edge_iterator> EdgeRange;
  typedef std::pair<NodeRange, EdgeRange> GraphRange;

  // numNodes = 0 corner case
  if (numNodes == 0) {
    return GraphRange(NodeRange(iterator(0), iterator(0)),
                      EdgeRange(edge_iterator(0), edge_iterator(0)));
  }

  assert(nodeWeight != 0 || edgeWeight != 0);
  assert(total >= 1);
  assert(id >= 0 && id < total);

  // weight of all data
  uint64_t weight = numNodes * nodeWeight + (numEdges + 1) * edgeWeight;
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
  //galois::gDebug("Unit ", id, " block ", blockLower, " to ",
  //               blockUpper, " ", blockLower * blockWeight, " ",
  //               blockUpper * blockWeight);

  uint64_t nodesLower;
  // use prefix sum to find node bounds
  if (blockLower == 0) {
    nodesLower = 0;
  } else {
    nodesLower = internal::findIndexPrefixSum(
        nodeWeight, edgeWeight, blockWeight * blockLower, 0, numNodes,
        edgePrefixSum, edgeOffset, nodeOffset);
  }

  uint64_t nodesUpper;
  nodesUpper = internal::findIndexPrefixSum(
      nodeWeight, edgeWeight, blockWeight * blockUpper, nodesLower, numNodes,
      edgePrefixSum, edgeOffset, nodeOffset);

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

  //galois::gDebug("Unit ", id, " nodes ", nodesLower, " to ",
  //               nodesUpper, " edges ", edgesLower, " ",
  //               edgesUpper);

  return GraphRange(
      NodeRange(iterator(nodesLower), iterator(nodesUpper)),
      EdgeRange(edge_iterator(edgesLower), edge_iterator(edgesUpper)));
}

// second internal namespace
namespace internal {

/**
 * Checks the begin/end node and number of units to split to for corner cases
 * (e.g. only one unit to split to, only 1 node, etc.).
 *
 * @param unitsToSplit number of units to split nodes among
 * @param beginNode Beginning of range
 * @param endNode End of range, non-inclusive
 * @param returnRanges vector to store result in
 * @returns true if a corner case was found (indicates that returnRanges has
 * been finalized)
 */
bool unitRangeCornerCaseHandle(uint32_t unitsToSplit, uint32_t beginNode,
                               uint32_t endNode,
                               std::vector<uint32_t>& returnRanges);

/**
 * Helper function used by determineUnitRangesGraph that consists of the main
 * loop over all units and calls to divide by node to determine the
 * division of nodes to units.
 *
 * Saves the ranges to an argument vector provided by the caller.
 *
 * @tparam GraphTy type of the graph object
 *
 * @param graph The graph object to get prefix sum information from
 * @param unitsToSplit number of units to split nodes among
 * @param beginNode Beginning of range
 * @param endNode End of range, non-inclusive
 * @param returnRanges Vector to store unit offsets for ranges in
 * @param nodeAlpha The higher the number, the more weight nodes have in
 * determining division of nodes (edges have weight 1).
 */
template <typename GraphTy>
void determineUnitRangesLoopGraph(GraphTy& graph, uint32_t unitsToSplit,
                                  uint32_t beginNode, uint32_t endNode,
                                  std::vector<uint32_t>& returnRanges,
                                  uint32_t nodeAlpha) {
  assert(beginNode != endNode);

  uint32_t numNodesInRange = endNode - beginNode;
  uint64_t numEdgesInRange =
      graph.edge_end(endNode - 1) - graph.edge_begin(beginNode);
  uint64_t edgeOffset = *graph.edge_begin(beginNode);

  returnRanges[0] = beginNode;
  std::vector<unsigned int> dummyScaleFactor;

  for (uint32_t i = 0; i < unitsToSplit; i++) {
    // determine division for unit i
    auto nodeSplits =
        divideNodesBinarySearch<GraphTy, uint32_t>(
            numNodesInRange, numEdgesInRange, nodeAlpha, 1, i, unitsToSplit,
            graph, dummyScaleFactor, edgeOffset, beginNode)
            .first;

    // i.e. if there are actually assigned nodes
    if (nodeSplits.first != nodeSplits.second) {
      if (i != 0) {
        assert(returnRanges[i] == *(nodeSplits.first) + beginNode);
      } else { // i == 0
        assert(returnRanges[i] == beginNode);
      }
      returnRanges[i + 1] = *(nodeSplits.second) + beginNode;
    } else {
      // unit assinged no nodes, copy last one
      returnRanges[i + 1] = returnRanges[i];
    }

    galois::gDebug("LoopGraph Unit ", i, " gets nodes ", returnRanges[i], " to ",
                   returnRanges[i + 1], ", num edges is ",
                   graph.edge_end(returnRanges[i + 1] - 1) -
                       graph.edge_begin(returnRanges[i]));
  }
}

/**
 * Helper function used by determineUnitRangesPrefixSum that consists of the
 * main loop over all units and calls to divide by node to determine the
 * division of nodes to units.
 *
 * Saves the ranges to an argument vector provided by the caller.
 *
 * @tparam VectorTy type of the prefix sum object
 *
 * @param prefixSum Holds prefix sum information
 * @param unitsToSplit number of units to split nodes among
 * @param beginNode Beginning of range
 * @param endNode End of range, non-inclusive
 * @param returnRanges Vector to store unit offsets for ranges in
 * @param nodeAlpha The higher the number, the more weight nodes have in
 * determining division of nodes (edges have weight 1).
 */
template <typename VectorTy>
void determineUnitRangesLoopPrefixSum(VectorTy& prefixSum,
                                      uint32_t unitsToSplit, uint32_t beginNode,
                                      uint32_t endNode,
                                      std::vector<uint32_t>& returnRanges,
                                      uint32_t nodeAlpha) {
  assert(beginNode != endNode);

  uint32_t numNodesInRange = endNode - beginNode;

  uint64_t numEdgesInRange;
  uint64_t edgeOffset;
  if (beginNode != 0) {
    numEdgesInRange = prefixSum[endNode - 1] - prefixSum[beginNode - 1];
    edgeOffset      = prefixSum[beginNode - 1];
  } else {
    numEdgesInRange = prefixSum[endNode - 1];
    edgeOffset      = 0;
  }

  returnRanges[0] = beginNode;
  std::vector<unsigned int> dummyScaleFactor;

  for (uint32_t i = 0; i < unitsToSplit; i++) {
    // determine division for unit i
    auto nodeSplits =
        divideNodesBinarySearch<VectorTy, uint32_t>(
            numNodesInRange, numEdgesInRange, nodeAlpha, 1, i, unitsToSplit,
            prefixSum, dummyScaleFactor, edgeOffset, beginNode)
            .first;

    // i.e. if there are actually assigned nodes
    if (nodeSplits.first != nodeSplits.second) {
      if (i != 0) {
        assert(returnRanges[i] == *(nodeSplits.first) + beginNode);
      } else { // i == 0
        assert(returnRanges[i] == beginNode);
      }
      returnRanges[i + 1] = *(nodeSplits.second) + beginNode;
    } else {
      // unit assinged no nodes
      returnRanges[i + 1] = returnRanges[i];
    }

    galois::gDebug("Unit ", i, " gets nodes ", returnRanges[i], " to ",
                   returnRanges[i + 1]);
  }
}

/**
 * Sanity checks a finalized unit range vector.
 *
 * @param unitsToSplit number of units to split nodes among
 * @param beginNode Beginning of range
 * @param endNode End of range, non-inclusive
 * @param returnRanges Ranges to sanity check
 */
void unitRangeSanity(uint32_t unitsToSplit, uint32_t beginNode,
                     uint32_t endNode, std::vector<uint32_t>& returnRanges);

} // namespace internal

/**
 * Determines node division ranges for all nodes in a graph and returns it in
 * an offset vector. (node ranges = assigned nodes that a particular unit
 * of execution should work on)
 *
 * Checks for corner cases, then calls the main loop function.
 *
 * ONLY CALL AFTER GRAPH IS CONSTRUCTED as it uses functions that assume
 * the graph is already constructed.
 *
 * @tparam GraphTy type of the graph object
 *
 * @param graph The graph object to get prefix sum information from
 * @param unitsToSplit number of units to split nodes among
 * @param nodeAlpha The higher the number, the more weight nodes have in
 * determining division of nodes (edges have weight 1).
 * @returns vector that indirectly specifies which units get which nodes
 */
template <typename GraphTy>
std::vector<uint32_t> determineUnitRangesFromGraph(GraphTy& graph,
                                                   uint32_t unitsToSplit,
                                                   uint32_t nodeAlpha = 0) {
  uint32_t totalNodes = graph.size();

  std::vector<uint32_t> returnRanges;
  returnRanges.resize(unitsToSplit + 1);

  // check corner cases
  if (internal::unitRangeCornerCaseHandle(unitsToSplit, 0, totalNodes,
                                          returnRanges)) {
    return returnRanges;
  }

  // no corner cases: onto main loop over nodes that determines
  // node ranges
  internal::determineUnitRangesLoopGraph(graph, unitsToSplit, 0, totalNodes,
                                         returnRanges, nodeAlpha);

  internal::unitRangeSanity(unitsToSplit, 0, totalNodes, returnRanges);

  return returnRanges;
}

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
 * @tparam GraphTy type of the graph object
 *
 * @param graph The graph object to get prefix sum information from
 * @param unitsToSplit number of units to split nodes among
 * @param beginNode Beginning of range
 * @param endNode End of range, non-inclusive
 * @param nodeAlpha The higher the number, the more weight nodes have in
 * determining division of nodes (edges have weight 1).
 * @returns vector that indirectly specifies which units get which nodes
 */
template <typename GraphTy>
std::vector<uint32_t>
determineUnitRangesFromGraph(GraphTy& graph, uint32_t unitsToSplit,
                             uint32_t beginNode, uint32_t endNode,
                             uint32_t nodeAlpha = 0) {
  std::vector<uint32_t> returnRanges;
  returnRanges.resize(unitsToSplit + 1);

  if (internal::unitRangeCornerCaseHandle(unitsToSplit, beginNode, endNode,
                                          returnRanges)) {
    return returnRanges;
  }

  // no corner cases: onto main loop over nodes that determines
  // node ranges
  internal::determineUnitRangesLoopGraph(graph, unitsToSplit, beginNode,
                                         endNode, returnRanges, nodeAlpha);

  internal::unitRangeSanity(unitsToSplit, beginNode, endNode, returnRanges);

  return returnRanges;
}

/**
 * Uses the divideByNode function (which is binary search based) to
 * divide nodes among units using a provided prefix sum.
 *
 * @tparam VectorTy type of the prefix sum object
 *
 * @param unitsToSplit number of units to split nodes among
 * @param edgePrefixSum A prefix sum of edges
 * @param nodeAlpha amount of weight to give to nodes when dividing work among
 * threads
 * @returns vector that indirectly specifies how nodes are split amongs units
 * of execution
 */
template <typename VectorTy>
std::vector<uint32_t> determineUnitRangesFromPrefixSum(uint32_t unitsToSplit,
                                                       VectorTy& edgePrefixSum,
                                                       uint32_t nodeAlpha = 0) {
  assert(unitsToSplit > 0);

  std::vector<uint32_t> nodeRanges;
  nodeRanges.resize(unitsToSplit + 1);

  nodeRanges[0] = 0;

  uint32_t numNodes = edgePrefixSum.size();
  // handle corner case TODO there are better ways to do this, i.e. call helper
  if (numNodes == 0) {
    nodeRanges[0] = 0;

    for (uint32_t i = 0; i < unitsToSplit; i++) {
      nodeRanges[i + 1] = 0;
    }
    return nodeRanges;
  }

  uint64_t numEdges = edgePrefixSum[numNodes - 1];

  for (uint32_t i = 0; i < unitsToSplit; i++) {
    auto nodeSplits =
        divideNodesBinarySearch<VectorTy, uint32_t>(
            numNodes, numEdges, nodeAlpha, 1, i, unitsToSplit, edgePrefixSum)
            .first;

    // i.e. if there are actually assigned nodes
    if (nodeSplits.first != nodeSplits.second) {
      if (i != 0) {
        assert(nodeRanges[i] == *(nodeSplits.first));
      } else { // i == 0
        assert(nodeRanges[i] == 0);
      }
      nodeRanges[i + 1] = *(nodeSplits.second);
    } else {
      // unit assinged no nodes
      nodeRanges[i + 1] = nodeRanges[i];
    }

    galois::gDebug("Unit ", i, " gets nodes ", nodeRanges[i], " to ",
                   nodeRanges[i + 1]);
  }

  return nodeRanges;
}

/**
 * Uses the divideByNode function (which is binary search based) to
 * divide nodes among units using a provided prefix sum. Provide a node range
 * so that the prefix sum is only calculated using that range.
 *
 * @tparam VectorTy type of the prefix sum object
 *
 * @param unitsToSplit number of units to split nodes among
 * @param edgePrefixSum A prefix sum of edges
 * @param beginNode Beginning of range
 * @param endNode End of range, non-inclusive
 * @param nodeAlpha amount of weight to give to nodes when dividing work among
 * threads
 * @returns vector that indirectly specifies how nodes are split amongs units
 * of execution
 */
template <typename VectorTy>
std::vector<uint32_t>
determineUnitRangesFromPrefixSum(uint32_t unitsToSplit, VectorTy& edgePrefixSum,
                                 uint32_t beginNode, uint32_t endNode,
                                 uint32_t nodeAlpha = 0) {
  std::vector<uint32_t> returnRanges;
  returnRanges.resize(unitsToSplit + 1);

  if (internal::unitRangeCornerCaseHandle(unitsToSplit, beginNode, endNode,
                                          returnRanges)) {
    return returnRanges;
  }

  // no corner cases: onto main loop over nodes that determines
  // node ranges
  internal::determineUnitRangesLoopPrefixSum(
      edgePrefixSum, unitsToSplit, beginNode, endNode, returnRanges, nodeAlpha);

  internal::unitRangeSanity(unitsToSplit, beginNode, endNode, returnRanges);

  return returnRanges;
}

} // end namespace graphs
} // end namespace galois
#endif
