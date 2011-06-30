/** Graph converter -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Dimitrios Prountzos <dprountz@cs.utexas.edu>
 */
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Galois.h"

#include "Galois/Graphs/Serialize.h"

#include <iostream>
#include <vector>

static const char* help =
  "[-dimacs2gr | -rmat2gr | -gr2dimacs] <input file> <output file>\n"
  "Converts graph formats.\n"
  "Default: converts dimacs to binary gr format (-dimacs2gr)\n";

void rmat2gr(const char *infilename, const char *outfilename) {
  typedef Galois::Graph::FirstGraph<void,int,true> Graph;
  typedef Graph::GraphNode GNode;
  Graph graph;

  std::ifstream infile(infilename);

  // Skip to first non-comment line
  while (true) {
    if (infile.peek() != '%') {
      break;
    }
    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  int nnodes, nedges;
  infile >> nnodes >> nedges;
  infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::vector<GNode> nodes;
  nodes.resize(nnodes);
  for (int i = 0; i < nnodes; ++i) {
    GNode n = graph.createNode(Galois::Graph::GraphUnit());
    nodes[i] = n;
    graph.addNode(n, Galois::Graph::NONE);
  }

  for (int edge_num = 0; edge_num < nedges; ++edge_num) {
    int cur_id, cur_edges;
    infile >> cur_id >> cur_edges;
    assert(0 <= cur_id && cur_id < nnodes);
    assert(cur_edges >= 0);
    for (int j = 0; j < cur_edges; ++j) {
      int neighbor_id, weight;
      infile >> neighbor_id >> weight;
      assert(0 <= neighbor_id && neighbor_id < nnodes);
      graph.addEdge(nodes[cur_id], nodes[neighbor_id], weight);
    }

    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes << " Edges: " << nedges << "\n";

  outputGraph(outfilename, graph);
}

void gr2dimacs(const char *infilename, const char *outfilename) {
  typedef Galois::Graph::LC_FileGraph<int, int> Graph;
  typedef Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);
  graph.emptyNodeData();

  int nnodes = 0;
  int nedges = 0;
  for (Graph::active_iterator i = graph.active_begin(), e = graph.active_end();
      i != e; ++i) {
    GNode src = *i;
    graph.getData(src) = nnodes++;
    nedges += std::distance(graph.neighbor_begin(*i), graph.neighbor_end(*i));
  }

  std::ofstream file(outfilename);
  file << "p sp " << nnodes << " " << nedges << "\n";
  for (Graph::active_iterator i = graph.active_begin(), e = graph.active_end();
      i != e; ++i) {
    GNode src = *i;
    for (Graph::neighbor_iterator j = graph.neighbor_begin(src),
        f = graph.neighbor_end(src); j != f; ++j) {
      GNode dst = *j;
      int weight = graph.getEdgeData(src, dst);
      file << "a " << graph.getData(src)
        << " " << graph.getData(dst)
        << " " << weight << "\n";
    }
  }
  file.close();

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes << " Edges: " << nedges 
    << "\n";

}

void dimacs2gr(const char *infilename, const char *outfilename) {
  typedef Galois::Graph::FirstGraph<void,int,true> Graph;
  typedef Graph::GraphNode GNode;
  Graph graph;

  std::ifstream infile(infilename);

  while (true) {
    if (infile.peek() != 'c') {
      break;
    }
    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  std::string tmp;
  int nnodes, nedges;
  infile >> tmp >> tmp >> nnodes >> nedges;
  infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::vector<GNode> nodes;
  nodes.resize(nnodes);
  for (int i = 0; i < nnodes; ++i) {
    GNode n = graph.createNode(Galois::Graph::GraphUnit());
    nodes[i] = n;
    graph.addNode(n, Galois::Graph::NONE);
  }

  for (int edge_num = 0; edge_num < nedges; ++edge_num) {
    int cur_id, neighbor_id, weight;
    infile >> tmp >> cur_id >> neighbor_id >> weight;
    assert(tmp.compare("a") == 0);
    assert(0 <= cur_id && cur_id < nnodes);
    assert(0 <= neighbor_id && neighbor_id < nnodes);
    graph.addEdge(nodes[cur_id], nodes[neighbor_id], weight);

    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes << " Edges: " << nedges << "\n";

  outputGraph(outfilename, graph);
}

int main(int argc, const char** argv) {
  int func = 0;
  for (int index = 1; index < argc; ++index) {
    const char* tok = argv[index];
    if (strcmp(tok, "-help") == 0) {
      std::cout << help;
      return 0;
    } else if (strcmp(tok, "-dimacs2gr") == 0) {
      func = 0;
    } else if (strcmp(tok, "-rmat2gr") == 0) {
      func = 1;
    } else if (strcmp(tok, "-gr2dimacs") == 0) {
      func = 2;
    } else if (argc - index == 2) {
      switch (func) {
        case 2: gr2dimacs(argv[index], argv[index+1]); break;
        case 1: rmat2gr(argv[index], argv[index+1]); break;
        case 0: dimacs2gr(argv[index], argv[index+1]); break;
        default: assert(false);
      }
      return 0;
    } else {
      std::cerr << "unknown arguments, use -help for usage information\n";
      return 1;
    }
  }
  std::cerr << help;
  return 1;
}
