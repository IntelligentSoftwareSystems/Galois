#include "Auxiliary.h"

#include <iostream>
#include <limits>

const size_t DIST_INFINITY = std::numeric_limits<size_t>::max() - 1;

NodeList createNodeList(int num) {
  GNode *l = NULL;
  if (num)
    l = new GNode [num] ();
  return {num, l};
}

void printNodeList(NodeList nl) {
  for (auto i = 0; i < nl.num; ++i) {
    std::cout << nl.nodes[i] << " ";
  }
  std::cout << std::endl;
}

void deleteNodeList(NodeList nl) {
  delete[] nl.nodes;
}

void deleteNodeDoubles(NodeDouble *array) {
  delete[] array;
}
void deleteGraphMatches(NodePair *pairs) {
  delete[] pairs;
}

