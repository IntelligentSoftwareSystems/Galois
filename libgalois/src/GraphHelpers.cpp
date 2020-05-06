/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include <galois/graphs/GraphHelpers.h>

namespace galois {
namespace graphs {
namespace internal {

uint32_t determine_block_division(uint32_t numDivisions,
                                  std::vector<unsigned>& scaleFactor) {
  uint32_t numBlocks = 0;

  if (scaleFactor.empty()) {
    // if scale factor isn't specified, everyone gets the same amount
    numBlocks = numDivisions;

    // scale factor holds a prefix sum of the scale factor
    for (uint32_t i = 0; i < numDivisions; i++) {
      scaleFactor.push_back(i + 1);
    }
  } else {
    assert(scaleFactor.size() == numDivisions);
    assert(numDivisions >= 1);

    // get numDivisions number of blocks we need + save a prefix sum of the
    // scale factor vector to scaleFactor
    for (uint32_t i = 0; i < numDivisions; i++) {
      numBlocks += scaleFactor[i];
      scaleFactor[i] = numBlocks;
    }
  }

  return numBlocks;
}

bool unitRangeCornerCaseHandle(uint32_t unitsToSplit, uint32_t beginNode,
                               uint32_t endNode,
                               std::vector<uint32_t>& returnRanges) {
  uint32_t totalNodes = endNode - beginNode;

  // check corner cases
  // no nodes = assign nothing to all units
  if (beginNode == endNode) {
    returnRanges[0] = beginNode;

    for (uint32_t i = 0; i < unitsToSplit; i++) {
      returnRanges[i + 1] = beginNode;
    }

    return true;
  }

  // single unit case; 1 unit gets all
  if (unitsToSplit == 1) {
    returnRanges[0] = beginNode;
    returnRanges[1] = endNode;
    return true;
    // more units than nodes
  } else if (unitsToSplit > totalNodes) {
    uint32_t current_node = beginNode;
    returnRanges[0]       = current_node;
    // 1 node for units until out of units
    for (uint32_t i = 0; i < totalNodes; i++) {
      returnRanges[i + 1] = ++current_node;
    }
    // deal with remainder units; they get nothing
    for (uint32_t i = totalNodes; i < unitsToSplit; i++) {
      returnRanges[i + 1] = totalNodes;
    }

    return true;
  }

  return false;
}

void unitRangeSanity(
    uint32_t GALOIS_USED_ONLY_IN_DEBUG(unitsToSplit),
    uint32_t GALOIS_USED_ONLY_IN_DEBUG(beginNode),
    uint32_t GALOIS_USED_ONLY_IN_DEBUG(endNode),
    std::vector<uint32_t>& GALOIS_USED_ONLY_IN_DEBUG(returnRanges)) {
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
}

} // namespace internal
} // namespace graphs
} // namespace galois
