#include "NodeList.h"

NodeList createNodeList(int num) {
  GNode *l = NULL;
  if (num)
    l = new GNode [num] ();
  return {num, l};
}

void deleteNodeList(NodeList nl) {
  delete[] nl.nodes;
}

