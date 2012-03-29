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
#include "Galois/Galois.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/Serialize.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <vector>
#include <string>

namespace cll = llvm::cl;

enum ConvertMode {
  dimacs2gr,
  rmat2gr,
  gr2bsml,
  gr2dimacs
};

static cll::opt<std::string> inputfilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputfilename(cll::Positional, cll::desc("<output file>"), cll::Required);
static cll::opt<ConvertMode> convertMode(cll::desc("Choose a conversion mode:"),
    cll::values(
      clEnumVal(dimacs2gr, "Convert dimacs to binary gr (default)"),
      clEnumVal(rmat2gr, "Convert rmat to binary gr"),
      clEnumVal(gr2bsml, "Convert binary gr to binary sparse MATLAB matrix"),
      clEnumVal(gr2dimacs, "Convert binary gr to dimacs"),
      clEnumValEnd), cll::init(dimacs2gr));

void convert_rmat2gr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FirstGraph<int,int,true> Graph;
  typedef Graph::GraphNode GNode;
  Graph graph;

  std::ifstream infile(infilename.c_str());

  // Skip to first non-comment line
  while (!infile.eof()) {
    if (infile.peek() != '%') {
      break;
    }
    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  size_t nnodes, nedges;
  infile >> nnodes >> nedges;
  infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::vector<GNode> nodes;
  nodes.resize(nnodes);
  for (size_t i = 0; i < nnodes; ++i) {
    GNode n = graph.createNode(i);
    nodes[i] = n;
    graph.addNode(n, Galois::NONE);
  }

  size_t edges_added = 0;
  for (size_t edge_num = 0; edge_num < nnodes; ++edge_num) {
    size_t cur_id, cur_edges;
    infile >> cur_id >> cur_edges;
    if (cur_id < 0 || cur_id >= nnodes) {
      std::cerr << "node id out of range: " << cur_id << "\n";
      abort();
    }
    if (cur_edges < 0) {
      std::cerr << "num edges out of range: " << cur_edges << "\n";
      abort();
    }
    
    for (size_t j = 0; j < cur_edges; ++j) {
      size_t neighbor_id, weight;
      infile >> neighbor_id >> weight;
      if (neighbor_id < 0 || neighbor_id >= nnodes) {
        std::cerr << "neighbor id out of range: " << neighbor_id << "\n";
        abort();
      }
      GNode& src = nodes[cur_id];
      GNode& dst = nodes[neighbor_id];
      if (src.hasNeighbor(dst)) {
        std::cerr << "Warning: Duplicate edge ("
          << cur_id << ", " << neighbor_id << ") weight " << weight
          << " ignored\n";
      } else {
        graph.addEdge(src, dst, weight);
        edges_added++;
      }
    }

    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  infile.peek();
  if (!infile.eof()) {
    std::cerr << "additional lines in file\n";
    abort();
  }

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes
    << " Edges read: " << nedges 
    << " Edges added: " << edges_added
    << "\n";

  outputGraph(outfilename.c_str(), graph);
}

void convert_dimacs2gr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FirstGraph<int,int,true> Graph;
  typedef Graph::GraphNode GNode;
  Graph graph;

  std::ifstream infile(infilename.c_str());

  while (!infile.eof()) {
    if (infile.peek() != 'c') {
      break;
    }
    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  if (infile.peek() != 'p') {
    std::cerr << "Missing problem specification line\n";
    abort();
  }

  std::string tmp;
  size_t nnodes, nedges;
  infile >> tmp >> tmp >> nnodes >> nedges;
  infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::vector<GNode> nodes;
  nodes.resize(nnodes);
  for (size_t i = 0; i < nnodes; ++i) {
    GNode n = graph.createNode(i);
    nodes[i] = n;
    graph.addNode(n, Galois::NONE);
  }

  size_t edges_added = 0;
  for (size_t edge_num = 0; edge_num < nedges; ++edge_num) {
    size_t cur_id, neighbor_id;
    int weight;
    infile >> tmp;

    if (tmp.compare("a") != 0) {
      --edge_num;
      infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }

    infile >> cur_id >> neighbor_id >> weight;
    if (cur_id == 0 || cur_id > nnodes) {
      std::cerr << "node id out of range: " << cur_id << "\n";
      abort();
    }
    if (neighbor_id == 0 || neighbor_id > nnodes) {
      std::cerr << "neighbor id out of range: " << neighbor_id << "\n";
      abort();
    }
    
    GNode& src = nodes[cur_id - 1]; // 1 indexed
    GNode& dst = nodes[neighbor_id - 1];
    if (src.hasNeighbor(dst)) {
      std::cerr << "Warning: Duplicate edge ("
        << cur_id << ", " << neighbor_id << ") weight " << weight
        << " ignored\n";
    } else {
      graph.addEdge(src, dst, weight);
      edges_added++;
    }

    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  infile.peek();
  if (!infile.eof()) {
    std::cerr << "additional lines in file\n";
    abort();
  }

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes
    << " Edges read: " << nedges 
    << " Edges added: " << edges_added
    << "\n";

  outputGraph(outfilename.c_str(), graph);
}

void convert_gr2dimacs(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::LC_CSR_Graph<size_t, int> Graph;
  typedef Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  size_t nnodes = 0;
  size_t nedges = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    graph.getData(src) = nnodes++;
    nedges += std::distance(graph.edge_begin(*ii), graph.edge_end(*ii));
  }

  std::ofstream file(outfilename.c_str());
  file << "p sp " << nnodes << " " << nedges << "\n";
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      int weight = graph.getEdgeData(jj);
      file << "a " << graph.getData(src) + 1
        << " " << graph.getData(dst) + 1
        << " " << weight << "\n";
    }
  }
  file.close();

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes << " Edges: " << nedges 
    << "\n";
}

/**
 * GR to Binary Sparse MATLAB matrix.
 * [i, j, v] = find(A); 
 * fwrite(f, size(A,1), 'uint32');
 * fwrite(f, size(A,2), 'uint32');
 * fwrite(f, nnz(A), 'uint32');
 * fwrite(f, (i-1), 'uint32');     % zero-indexed
 * fwrite(f, (j-1), 'uint32'); 
 * fwrite(f, v, 'double');
 */
void convert_gr2bsml(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::LC_CSR_Graph<uint32_t, int> Graph;
  typedef Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  uint32_t nnodes = 0;
  uint32_t nedges = 0;
  for (Graph::iterator i = graph.begin(), e = graph.end();
      i != e; ++i) {
    GNode src = *i;
    graph.getData(src) = nnodes++;
    nedges += std::distance(graph.edge_begin(*i), graph.edge_end(*i));
  }

  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  int fd = open(outfilename.c_str(), O_WRONLY | O_CREAT |O_TRUNC, mode);
  int retval;

  // Write header
  retval = write(fd, &nnodes, sizeof(nnodes));
  if (retval == -1) { perror(__FILE__); abort(); }
  retval = write(fd, &nnodes, sizeof(nnodes));
  if (retval == -1) { perror(__FILE__); abort(); }
  retval = write(fd, &nedges, sizeof(nedges));
  if (retval == -1) { perror(__FILE__); abort(); }

  // Write row adjacency
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    uint32_t sid = graph.getData(src);
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      retval = write(fd, &sid, sizeof(sid));
      if (retval == -1) { perror(__FILE__); abort(); }
    }
  }

  // Write column adjacency
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      uint32_t did = graph.getData(dst);
      retval = write(fd, &did, sizeof(did));
      if (retval == -1) { perror(__FILE__); abort(); }
    }
  }

  // Write data
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      //GNode dst = graph.getEdgeDst(jj);
      double weight = graph.getEdgeData(jj);
      retval = write(fd, &weight, sizeof(weight));
      if (retval == -1) { perror(__FILE__); abort(); }
    }
  }

  close(fd);

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes << " Edges: " << nedges 
    << "\n";
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  switch (convertMode) {
    case rmat2gr: convert_rmat2gr(inputfilename, outputfilename); break;
    case gr2dimacs: convert_gr2dimacs(inputfilename, outputfilename); break;
    case gr2bsml: convert_gr2bsml(inputfilename, outputfilename); break;
    default:
    case dimacs2gr: convert_dimacs2gr(inputfilename, outputfilename); break;
  }
  return 0;
}
