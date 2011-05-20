// Graph (.gr files) -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

/*  A graph contains n nodes and m arcs */
/*  Nodes are identified by integers 1...n */
/*  Graphs can be interpreted as directed or undirected, depending on the problem being studied */
/*  Graphs can have parallel arcs and self-loops */
/*  Arc weights are signed integers */

/* Comment lines:  Comment lines can appear anywhere and are ignored by programs. */
/* c This is a comment */
/* Problem line: The problem line is unique and must appear as the first non-comment line. This line has the format on the right, where n and m are the number of nodes and the number of arcs, respectively.	 */
/* p sp n m */
/* Arc descriptor lines: Arc descriptors are of the form on the right, where U and V are the tail and the head node ids, respectively, and W is the arc weight. */
/* a U V W */

#include <iostream>
#include <fstream>
#include <string>

namespace Galois {
namespace IO {

struct UnitWeights {
  template<typename GraphTy, typename GraphNodeTy>
  static void addEdge(GraphTy* graph, GraphNodeTy src, GraphNodeTy dst, int weight) {
    graph->addEdge(src,dst, 1, Galois::Graph::NONE);
  } 
};

struct ReadWeights {
  template<typename GraphTy, typename GraphNodeTy>
  static void addEdge(GraphTy* graph, GraphNodeTy src, GraphNodeTy dst, int weight) {
    graph->addEdge(src,dst, weight, Galois::Graph::NONE);
  } 
};

struct VoidWeights {
  template<typename GraphTy, typename GraphNodeTy>
  static void addEdge(GraphTy* graph, GraphNodeTy src, GraphNodeTy dst, int weight) {
    graph->addEdge(src,dst, Galois::Graph::NONE);
  } 
};


template<typename GraphTy, typename WeightTy>
std::pair<unsigned int, unsigned int> readFile_gr_(const char *filename, GraphTy* graph, WeightTy ae) {
  std::ifstream infile;
  infile.open(filename, std::ifstream::in); // opens the vector file
  if (!infile) { // file couldn't be opened
    std::cerr << "vector file could not be opened\n";
    abort();
  }
  
  std::string name;
  unsigned int LEdgeCount = 0;
  unsigned int numNodes = 0;
  unsigned int numEdges = 0;
  typename GraphTy::GraphNode* gnodes = NULL;
  while (!infile.eof()) {
    char firstchar = 0;
    infile >> firstchar;
    if (infile.fail()) {
      break;
    } else if (firstchar == 'a') {
      ++LEdgeCount;
      int src, dest, weight;
      infile >> src >> dest >> weight;
      assert(weight > 0);
      WeightTy::addEdge(graph, gnodes[src - 1], gnodes[dest - 1], weight);
    } else if (firstchar == 'c') {
      std::string line;
      getline(infile, line);
    } else if (firstchar == 'p') {
      infile >> name;
      infile >> numNodes;
      infile >> numEdges;
      gnodes = new typename GraphTy::GraphNode[numNodes];
      for (unsigned int i = 0; i < numNodes; i++) {
	gnodes[i] = graph->createNode(i+1);
	graph->addNode(gnodes[i], Galois::Graph::NONE);
      }
    } else {
      std::cerr << "Failure reading file " << filename
		<< " because of line begining with " << firstchar
		<< "\n";
      abort();
    } 
  }
  delete[] gnodes;
  if (LEdgeCount != numEdges) {
    std::cerr << "Read edges " << LEdgeCount << " don't match claimed " << numEdges << "\n";
    assert(LEdgeCount == numEdges);
    abort();
  }
  infile.close();

  return std::make_pair((unsigned)numNodes, (unsigned)numEdges);
}

template<typename GraphTy>
std::pair<unsigned int, unsigned int> readFile_gr(const char *filename, GraphTy* graph) {
  return readFile_gr_(filename, graph, ReadWeights());
}

template<typename GraphTy>
std::pair<unsigned int, unsigned int> readFile_gr_unit(const char *filename, GraphTy* graph) {
  return readFile_gr_(filename, graph, UnitWeights());
}

template<typename GraphTy>
std::pair<unsigned int, unsigned int> readFile_gr_void(const char *filename, GraphTy* graph) {
  return readFile_gr_(filename, graph, VoidWeights());
}



}
}
