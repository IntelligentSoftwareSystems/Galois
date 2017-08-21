/** Graph helper functions -*- C++ -*-
 * @file GraphHelpers.h
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Contains functions that can be done on various graphs with a particular
 * interface. Factored out from FileGraph, OfflineGraph, LC_CSR_Graph.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef GALOIS_GRAPH_HELPERS
#define GALOIS_GRAPH_HELPERS

#include <boost/iterator/counting_iterator.hpp>

namespace Galois {
namespace Graph {

/**
 * Return a suitable index between an upper bound and a lower bound that
 * attempts to get close to the target size (i.e. find a good chunk that
 * corresponds to some size). 
 *
 * @tparam GraphClass class of the input graph; should have edge_end function
 * @tparam NodeType type of node data
 *
 * @param inputGraph graph to find index in
 * @param nodeWeight weight to give to a node in division
 * @param edgeWeight weight to give to an edge in division
 * @param targetWeight The amount of weight we want from the returned index
 * @param lb lower bound to start search from
 * @param ub upper bound to start search from
 * @param nodeOffset number of nodes to subtract (offset)
 * @param edgeOffset number of edges to subtract (offset)
 */
template <typename GraphClass, typename NodeType>
inline uint64_t findIndex(GraphClass& inputGraph, size_t nodeWeight, size_t edgeWeight, 
                 uint64_t targetWeight, uint64_t lb, uint64_t ub,
                 NodeType nodeOffset = 0, uint64_t edgeOffset = 0) {
  assert(nodeWeight != 0 || edgeWeight != 0);

  while (lb < ub) {
    uint64_t mid = lb + (ub - lb) / 2;
    uint64_t num_edges = *(inputGraph.edge_begin(mid)) - edgeOffset;
    uint64_t weight = (num_edges * edgeWeight) + ((mid - nodeOffset) * nodeWeight);

    if (weight < targetWeight)
      lb = mid + 1;
    else if (weight == targetWeight)
      lb = mid + 1;
    else
      ub = mid;
  }

  return lb;
}

/**
 * Return a suitable index between an upper bound and a lower bound that
 * attempts to get close to the target size (i.e. find a good chunk that
 * corresponds to some size) using a prefix sum.
 *
 * @param nodeWeight weight to give to a node in division
 * @param edgeWeight weight to give to an edge in division
 * @param targetWeight The amount of weight we want from the returned index
 * @param lb lower bound to start search from
 * @param ub upper bound to start search from
 * @param edgePrefixSum prefix sum of edges
 */
inline size_t findIndexPrefixSum(size_t nodeWeight, size_t edgeWeight, 
                          size_t targetWeight, size_t lb, size_t ub, 
                          std::vector<uint64_t> edgePrefixSum) {
  assert(nodeWeight != 0 || edgeWeight != 0);

  while (lb < ub) {
    size_t mid = lb + (ub - lb) / 2;
    size_t num_edges;

    if (mid != 0) {
      num_edges = edgePrefixSum[mid - 1];
    } else {
      num_edges = 0;
    }

    size_t weight = num_edges * edgeWeight + (mid) * nodeWeight;

    if (weight <= targetWeight)
      lb = mid + 1;
    else if (weight == targetWeight)
      lb = mid + 1;
    else
      ub = mid;
  }
  return lb;
}



/** 
 * Returns 2 ranges (one for nodes, one for edges) for a particular division.
 * The ranges specify the nodes/edges that a division is responsible for. The
 * function attempts to split them evenly among threads given some kind of
 * weighting
 *
 * @tparam GraphClass type of the input graph
 * @tparam NodeType type of node data

 * @param nodeWeight weight to give to a node in division
 * @param edgeWeight weight to give to an edge in division
 * @param id Division number you want the ranges for
 * @param total Total number of divisions
 * @param scaleFactor Vector specifying if certain divisions should get more 
 * than other divisions
 * @param nodeOffset number of nodes to subtract (offset)
 * @param edgeOffset number of edges to subtract (offset)
 */
template <typename GraphClass, typename NodeType = uint32_t> inline
std::pair<std::pair<boost::counting_iterator<NodeType>,
                    boost::counting_iterator<NodeType>>,
          std::pair<boost::counting_iterator<uint64_t>,
                    boost::counting_iterator<uint64_t>>>
divideNodesBinarySearch(GraphClass& inputGraph, 
                   size_t nodeWeight, 
                   size_t edgeWeight, 
                   size_t id, 
                   size_t total,
                   std::vector<unsigned> scaleFactor = std::vector<unsigned>(),
                   NodeType nodeOffset = 0, 
                   uint64_t edgeOffset = 0,
                   std::vector<uint64_t> edgePrefixSum = std::vector<uint64_t>())
{ 
  typedef boost::counting_iterator<NodeType> iterator;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef std::pair<iterator, iterator> NodeRange;
  typedef std::pair<edge_iterator, edge_iterator> EdgeRange;
  typedef std::pair<NodeRange, EdgeRange> GraphRange;

  assert(nodeWeight != 0 || edgeWeight != 0);

  assert(total >= 1);
  assert(id >= 0 && id < total);


  NodeType numNodes = inputGraph.size();
  NodeType numEdges = inputGraph.sizeEdges();

  // weight of all data
  uint64_t weight = numNodes * nodeWeight + numEdges * edgeWeight;

  // determine number of blocks to divide among total divisions
  uint32_t numBlocks = 0;
  if (scaleFactor.empty()) {
    numBlocks = total;

    // scale factor holds a prefix sum of the scale factor
    for (uint32_t i = 0; i < total; i++) {
      scaleFactor.push_back(i + 1);
    }
  } else {
    assert(scaleFactor.size() == total);
    assert(total >= 1);

    // get total number of blocks we need + save a prefix sum of the scale
    // factor vector
    for (uint32_t i = 0; i < total; i++) {
      numBlocks += scaleFactor[i];
      scaleFactor[i] = numBlocks;
    }
  }

  // weight of a block (one block for each division by default; if scale
  // factor specifies something different, then use that instead)
  uint64_t blockWeight = (weight + numBlocks - 1) / numBlocks;

  // lower and upper blocks that this division should get using the prefix
  // sum of scaleFactor calculated above
  uint32_t blockLower;
  if (id != 0) {
    blockLower = scaleFactor[id - 1];
  } else {
    blockLower = 0;
  }
  uint32_t blockUpper = scaleFactor[id];

  assert(blockLower <= blockUpper);

  uint64_t nodesLower;
  uint64_t nodesUpper;
  uint64_t edgesLower = numEdges;
  uint64_t edgesUpper = numEdges;

  if (edgePrefixSum.size() == 0) {
    // find allocation of nodes for this division
    if (blockLower == 0) {
      nodesLower = 0;
    } else {
      nodesLower = findIndex(inputGraph, nodeWeight, edgeWeight, 
                             blockWeight * blockLower, 0, numNodes,
                             nodeOffset, edgeOffset);
    }

    nodesUpper = findIndex(inputGraph, nodeWeight, edgeWeight,
                           blockWeight * blockUpper, nodesLower,
                           numNodes, nodeOffset, edgeOffset);

    // correct number of edges based on nodes allocated to division if
    // necessary
    if (nodesLower != nodesUpper) {
      edgesLower = *(inputGraph.edge_begin(nodesLower));
      edgesUpper = *(inputGraph.edge_end(nodesUpper - 1));
    }
  } else {
    // use prefix sums
    assert((uint64_t)edgePrefixSum.size() == numNodes);

    if (blockLower == 0) {
      nodesLower = 0;
    } else {
      nodesLower = findIndexPrefixSum(nodeWeight, edgeWeight, 
                                      blockWeight * blockLower, 0, numNodes, 
                                      edgePrefixSum);
    }
    nodesUpper = findIndexPrefixSum(nodeWeight, edgeWeight, 
                                    blockWeight * blockUpper, nodesLower, 
                                    numNodes, edgePrefixSum);
 
    // correct number of edges
    if (nodesLower != nodesUpper) {
      if (nodesLower != 0) {
        edgesLower = edgePrefixSum[nodesLower - 1];
      } else {
        edgesLower = 0;
      }
      edgesUpper = edgePrefixSum[nodesUpper - 1];
    }
  }

  return GraphRange(NodeRange(iterator(nodesLower), 
                              iterator(nodesUpper)), 
                    EdgeRange(edge_iterator(edgesLower), 
                              edge_iterator(edgesUpper)));
}


} // end namespace Graph
} // end namespace Galois
#endif
