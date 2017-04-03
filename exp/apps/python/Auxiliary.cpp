#include "Auxiliary.h"

#include <iostream>
#include <limits>

const size_t DIST_INFINITY = std::numeric_limits<size_t>::max() - 1;

static AttrList makeAttrCopy(Attr& attr) {
  size_t num = attr.size();

  KeyAltTy *key = nullptr; 
  ValAltTy *value = nullptr;

  if (num) {
    key = new KeyAltTy [num] ();
    value = new ValAltTy [num] ();

    size_t i = 0;
    for (auto k: attr) {
      // deep copy for strings
      key[i] = new std::string::value_type [k.first.size()+1] ();
      std::copy(k.first.begin(), k.first.end(), key[i]);
      value[i] = new std::string::value_type [k.second.size()+1] ();
      std::copy(k.second.begin(), k.second.end(), value[i]);
      i++;
    }
  }

  return {num, key, value};
}

AttrList getNodeAllAttr(Graph *g, GNode n) {
  return makeAttrCopy(g->getData(n).attr);
}

AttrList getEdgeAllAttr(Graph *g, Edge e) {
  auto ei = g->findEdge(e.src, e.dst);
  assert(ei != g->edge_end(e.src));
  return makeAttrCopy(g->getEdgeData(ei));
}

NodeList getAllNodes(Graph *g) {
  NodeList l = createNodeList(std::distance(g->begin(), g->end()));
  auto i = 0;
  for (auto n: *g) {
    l.nodes[i++] = n;
  }
  return l;
}

EdgeList getAllEdges(Graph *g) {
  size_t num = 0;
  for (auto n: *g) {
    num += std::distance(g->edge_begin(n), g->edge_end(n));
  }

  EdgeList l = createEdgeList(num);
  auto i = 0;
  for (auto n: *g) {
    for (auto e: g->edges(n)) {
      l.edges[i].src = n;
      l.edges[i].dst = g->getEdgeDst(e);
      i++;
    }
  }
  return l; 
}

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

EdgeList createEdgeList(int num) {
  Edge *l = NULL;
  if (num)
    l = new Edge [num] ();
  return {num, l};
}

void printEdgeList(EdgeList el) {
  for (auto i = 0; i < el.num; ++i) {
    std::cout << "(" << el.edges[i].src << ", " << el.edges[i].dst << ") ";
  }
  std::cout << std::endl;
}

void deleteEdgeList(EdgeList el) {
  delete[] el.edges;
}

void deleteNodeDoubles(NodeDouble *array) {
  delete[] array;
}

void deleteGraphMatches(NodePair *pairs) {
  delete[] pairs;
}

void deleteAttrList(AttrList l) {
  if (0 == l.num) {
    return;
  }

  for (int i = 0; i < l.num; ++i) {
    delete[] l.key[i];
    delete[] l.value[i];
  }

  delete[] l.key;
  delete[] l.value;
}

