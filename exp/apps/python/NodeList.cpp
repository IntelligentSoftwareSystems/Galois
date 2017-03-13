#include "NodeList.h"

#include <iostream>

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

