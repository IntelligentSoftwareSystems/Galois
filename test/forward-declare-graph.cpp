#include "galois/graphs/Graph.h"
#include "galois/graphs/LCGraph.h"

struct Node1;
typedef galois::graphs::MorphGraph<Node1, void, true> Graph1;
struct Node1 {
  Graph1::edge_iterator edge;
  Graph1::GraphNode gnode;
};

struct Node2;
typedef galois::graphs::LC_CSR_Graph<Node2, void> Graph2;
struct Node2 {
  Graph2::edge_iterator edge;
  Graph2::GraphNode gnode;
};

struct Node3;
typedef galois::graphs::LC_InlineEdge_Graph<Node3, void> Graph3;
struct Node3 {
  Graph3::edge_iterator edge;
  Graph3::GraphNode gnode;
};

struct Node4;
typedef galois::graphs::LC_Linear_Graph<Node4, void> Graph4;
struct Node4 {
  Graph4::edge_iterator edge;
  Graph4::GraphNode gnode;
};

struct Node5;
typedef galois::graphs::LC_Morph_Graph<Node5, void> Graph5;
struct Node5 {
  Graph5::edge_iterator edge;
  Graph5::GraphNode gnode;
};

int main() {
  Graph1 g1;
  Graph2 g2;
  Graph3 g3;
  Graph4 g4;
  Graph5 g5;
  return 0;
}
