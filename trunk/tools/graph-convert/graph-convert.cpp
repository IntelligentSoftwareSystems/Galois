/** Graph converter -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
#include "Galois/Graphs/Graph2.h"
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
  stext2vgr,
  pbbs2vgr,
  pbbs2gr,
  gr2bsml,
  vgr2bsml,
  gr2dimacs,
  gr2pbbs
};

static cll::opt<std::string> inputfilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputfilename(cll::Positional, cll::desc("<output file>"), cll::Required);
static cll::opt<ConvertMode> convertMode(cll::desc("Choose a conversion mode:"),
    cll::values(
      clEnumVal(dimacs2gr, "Convert dimacs to binary gr (default)"),
      clEnumVal(rmat2gr, "Convert rmat to binary gr"),
      clEnumVal(pbbs2vgr, "Convert pbbs graph to binary void gr"),
      clEnumVal(stext2vgr, "Convert simple text graph to binary void gr"),
      clEnumVal(pbbs2gr, "Convert pbbs graph to binary gr"),
      clEnumVal(gr2bsml, "Convert binary gr to binary sparse MATLAB matrix"),
      clEnumVal(vgr2bsml, "Convert binary void gr to binary sparse MATLAB matrix"),
      clEnumVal(gr2dimacs, "Convert binary gr to dimacs"),
      clEnumVal(gr2pbbs, "Convert binary gr to pbbs"),
      clEnumValEnd), cll::init(dimacs2gr));

void convert_rmat2gr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FirstGraph<uint32_t,int32_t,true> Graph;
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

  uint32_t nnodes;
  size_t nedges;
  infile >> nnodes >> nedges;
  infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::vector<GNode> nodes;
  nodes.resize(nnodes);
  for (uint32_t i = 0; i < nnodes; ++i) {
    GNode n = graph.createNode(i);
    graph.addNode(n);
    nodes[i] = n;
    graph.addNode(n, Galois::NONE);
  }

  for (size_t edge_num = 0; edge_num < nnodes; ++edge_num) {
    uint32_t cur_id;
    size_t cur_edges;
    infile >> cur_id >> cur_edges;
    if (cur_id >= nnodes) {
      std::cerr << "Error: node id out of range: " << cur_id << "\n";
      abort();
    }
    
    for (size_t j = 0; j < cur_edges; ++j) {
      uint32_t neighbor_id;
      int32_t weight;
      infile >> neighbor_id >> weight;
      if (neighbor_id >= nnodes) {
        std::cerr << "Error: neighbor id out of range: " << neighbor_id << "\n";
        abort();
      }
      GNode& src = nodes[cur_id];
      GNode& dst = nodes[neighbor_id];
      // replaces existing edge if it exists
      graph.getEdgeData(graph.addEdge(src, dst)) = weight;
    }

    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  infile.peek();
  if (!infile.eof()) {
    std::cerr << "Error: additional lines in file\n";
    abort();
  }

  size_t edges_added = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    edges_added += std::distance(graph.edge_begin(*ii), graph.edge_end(*ii));
  }

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes
    << " Edges read: " << nedges 
    << " Edges added: " << edges_added
    << "\n";

  outputGraph(outfilename.c_str(), graph);
}

/**
 * Simple undirected graph input.
 *
 * <num_nodes> <num_edges>
 * <neighbor 0 of node 0> <neighbor 1 of node 0> ...
 * <neighbor 0 of node 1> ...
 * ...
 */
void convert_stext2vgr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FirstGraph<uint32_t,void,true> Graph;
  typedef Graph::GraphNode GNode;
  Graph graph;

  std::ifstream infile(infilename.c_str());

  std::string tmp;
  uint32_t nnodes;
  size_t nedges;
  infile >> nnodes >> nedges;
  infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::vector<GNode> nodes;
  nodes.resize(nnodes);
  for (uint32_t i = 0; i < nnodes; ++i) {
    GNode n = graph.createNode(i);
    graph.addNode(n);
    nodes[i] = n;
    graph.addNode(n, Galois::NONE);
  }

  size_t edges_added = 0;
  for (uint32_t node_id = 0; node_id < nnodes; ++node_id) {
    GNode& src = nodes[node_id];
    while (true) {
      if (!infile.good()) {
        if (node_id != nnodes - 1) {
          std::cerr << "Error: read data until node " << node_id << " of " << nnodes << "\n";
          return;
        }
        break;
      }

      char c;
      infile.get(c);
      if (c == '\n')
        break;
      if (isspace(c))
        continue;
      
      infile.unget();
      uint32_t neighbor_id;
      infile >> neighbor_id;
      assert(neighbor_id < nnodes);
      GNode& dst = nodes[neighbor_id];
      graph.addEdge(src, dst);
      ++edges_added;
    }
  }

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes
    << " Edges read: " << nedges 
    << " Edges added: " << edges_added
    << "\n";

  outputGraph(outfilename.c_str(), graph);
}

void convert_dimacs2gr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FirstGraph<uint32_t,int32_t,true> Graph;
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
    std::cerr << "Error: missing problem specification line\n";
    abort();
  }

  std::string tmp;
  uint32_t nnodes;
  size_t nedges;
  infile >> tmp >> tmp >> nnodes >> nedges;
  infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::vector<GNode> nodes;
  nodes.resize(nnodes);
  for (uint32_t i = 0; i < nnodes; ++i) {
    GNode n = graph.createNode(i);
    graph.addNode(n);
    nodes[i] = n;
    graph.addNode(n, Galois::NONE);
  }

  for (size_t edge_num = 0; edge_num < nedges; ++edge_num) {
    uint32_t cur_id, neighbor_id;
    int32_t weight;
    infile >> tmp;

    if (tmp.compare("a") != 0) {
      --edge_num;
      infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }

    infile >> cur_id >> neighbor_id >> weight;
    if (cur_id == 0 || cur_id > nnodes) {
      std::cerr << "Error: node id out of range: " << cur_id << "\n";
      abort();
    }
    if (neighbor_id == 0 || neighbor_id > nnodes) {
      std::cerr << "Error: neighbor id out of range: " << neighbor_id << "\n";
      abort();
    }
    
    GNode& src = nodes[cur_id - 1]; // 1 indexed
    GNode& dst = nodes[neighbor_id - 1];
    // replaces existing edge if it exists
    graph.getEdgeData(graph.addEdge(src, dst)) = weight;

    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  infile.peek();
  if (!infile.eof()) {
    std::cerr << "Error: additional lines in file\n";
    abort();
  }

  size_t edges_added = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    edges_added += std::distance(graph.edge_begin(*ii), graph.edge_end(*ii));
  }

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes
    << " Edges read: " << nedges 
    << " Edges added: " << edges_added
    << "\n";

  outputGraph(outfilename.c_str(), graph);
}

/**
 * PBBS input is an ASCII file of tokens that serialize a CSR graph. I.e., 
 * elements in brackets are non-literals:
 * 
 * AdjacencyGraph
 * <num nodes>
 * <num edges>
 * <offset node 0>
 * <offset node 1>
 * ...
 * <edge 0>
 * <edge 1>
 * ...
 */
template<typename EdgeTy>
void convert_pbbs2gr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FirstGraph<uint32_t,EdgeTy,true> Graph;
  typedef typename Graph::GraphNode GNode;
  Graph graph;

  std::ifstream infile(infilename.c_str());
  std::string header;
  uint32_t nnodes;
  size_t nedges;

  infile >> header >> nnodes >> nedges;
  if (header != "AdjacencyGraph") {
    std::cerr << "Error: unknown file format\n";
    abort();
  }

  size_t* offsets = new size_t[nnodes];
  for (size_t i = 0; i < nnodes; ++i) {
    infile >> offsets[i];
  }

  size_t* edges = new size_t[nedges];
  for (size_t i = 0; i < nedges; ++i) {
    infile >> edges[i];
  }

  std::vector<GNode> nodes;
  nodes.resize(nnodes);
  for (uint32_t i = 0; i < nnodes; ++i) {
    GNode n = graph.createNode(i);
    graph.addNode(n);
    nodes[i] = n;
    graph.addNode(n, Galois::NONE);
  }

  for (uint32_t i = 0; i < nnodes; ++i) {
    size_t begin = offsets[i];
    size_t end = (i == nnodes - 1) ? nedges : offsets[i+1];
    GNode& src = nodes[i];
    for (size_t j = begin; j < end; ++j) {
      GNode& dst = nodes[edges[j]];
      graph.addEdge(src, dst);
    }
  }

  std::cout << "Finished reading graph. "
    << "Nodes: " << nnodes
    << " Edges read: " << nedges 
    << "\n";

  outputGraph(outfilename.c_str(), graph);
}

void convert_gr2pbbs(const std::string& infilename, const std::string& outfilename) {
  // Use FileGraph because it is basically in CSR format needed for pbbs
  typedef Galois::Graph::FileGraph Graph;

  Graph graph;
  graph.structureFromFile(infilename);

  std::ofstream file(outfilename.c_str());
  file << "AdjacencyGraph\n" << graph.size() << "\n" << graph.sizeEdges() << "\n";
  // edgeid[i] is the end of i in FileGraph while it is the beginning of i in pbbs graph
  size_t last = std::distance(graph.edgeid_begin(), graph.edgeid_end());
  size_t count = 0;
  file << "0\n";
  for (Graph::edgeid_iterator ii = graph.edgeid_begin(), ei = graph.edgeid_end();
      ii != ei; ++ii, ++count) {
    if (count < last - 1)
      file << *ii << "\n";
  }
  for (Graph::nodeid_iterator ii = graph.nodeid_begin(), ei = graph.nodeid_end(); ii != ei; ++ii) {
    file << *ii << "\n";
  }
  file.close();

  std::cout << "Finished reading graph. "
    << "Nodes: " << graph.size() << " Edges: " << graph.sizeEdges() 
    << "\n";
}

void convert_gr2dimacs(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::LC_CSR_Graph<uint32_t,int32_t> Graph;
  typedef Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  uint32_t nnodes = 0;
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
      int32_t weight = graph.getEdgeData(jj);
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
template<typename EdgeTy>
void convert_gr2bsml(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::LC_CSR_Graph<uint32_t,EdgeTy> Graph;
  typedef typename Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  uint32_t nnodes = 0;
  uint32_t nedges = 0;
  for (typename Graph::iterator i = graph.begin(), e = graph.end();
      i != e; ++i) {
    GNode src = *i;
    graph.getData(src) = nnodes++;
    nedges += std::distance(graph.edge_begin(*i), graph.edge_end(*i));
  }

  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  int fd = open(outfilename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
  int retval;

  // Write header
  retval = write(fd, &nnodes, sizeof(nnodes));
  if (retval == -1) { perror(__FILE__); abort(); }
  retval = write(fd, &nnodes, sizeof(nnodes));
  if (retval == -1) { perror(__FILE__); abort(); }
  retval = write(fd, &nedges, sizeof(nedges));
  if (retval == -1) { perror(__FILE__); abort(); }

  // Write row adjacency
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    uint32_t sid = graph.getData(src);
    for (typename Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      retval = write(fd, &sid, sizeof(sid));
      if (retval == -1) { perror(__FILE__); abort(); }
    }
  }

  // Write column adjacency
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (typename Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      uint32_t did = graph.getData(dst);
      retval = write(fd, &did, sizeof(did));
      if (retval == -1) { perror(__FILE__); abort(); }
    }
  }

  // Write data
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (typename Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
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
    case stext2vgr: convert_stext2vgr(inputfilename, outputfilename); break;
    case pbbs2vgr: convert_pbbs2gr<void>(inputfilename, outputfilename); break;
    case pbbs2gr: convert_pbbs2gr<int32_t>(inputfilename, outputfilename); break;
    case gr2dimacs: convert_gr2dimacs(inputfilename, outputfilename); break;
    case vgr2bsml: convert_gr2bsml<void>(inputfilename, outputfilename); break;
    case gr2bsml: convert_gr2bsml<int32_t>(inputfilename, outputfilename); break;
    case gr2pbbs: convert_gr2pbbs(inputfilename, outputfilename); break;
    default:
    case dimacs2gr: convert_dimacs2gr(inputfilename, outputfilename); break;
  }
  return 0;
}
