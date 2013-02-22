/** Graph converter -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <vector>
#include <string>

#include <fcntl.h>

namespace cll = llvm::cl;

enum ConvertMode {
  dimacs2gr,
  edgelist2gr,
  floatedgelist2gr,
  gr2bsml,
  gr2dimacs,
  gr2pbbs,
  gr2floatpbbs,
  gr2intpbbs,
  intedgelist2gr,
  pbbs2gr,
  pbbs2vgr,
  rmat2gr,
  vgr2bsml,
  vgr2cvgr,
  vgr2edgelist,
  vgr2svgr
};

static cll::opt<std::string> inputfilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputfilename(cll::Positional, cll::desc("<output file>"), cll::Required);
static cll::opt<ConvertMode> convertMode(cll::desc("Choose a conversion mode:"),
    cll::values(
      clEnumVal(dimacs2gr, "Convert dimacs to binary gr"),
      clEnumVal(edgelist2gr, "Convert edge list to binary gr (default)"),
      clEnumVal(floatedgelist2gr, "Convert weighted (float) edge list to binary gr"),
      clEnumVal(gr2bsml, "Convert binary gr to binary sparse MATLAB matrix"),
      clEnumVal(gr2dimacs, "Convert binary gr to dimacs"),
      clEnumVal(gr2pbbs, "Convert binary gr to pbbs"),
      clEnumVal(gr2floatpbbs, "Convert binary gr to weighted (float) pbbs"),
      clEnumVal(gr2intpbbs, "Convert binary gr to weighted (int) pbbs"),
      clEnumVal(intedgelist2gr, "Convert weighted (int) edge list to binary gr (default)"),
      clEnumVal(pbbs2gr, "Convert pbbs graph to binary gr"),
      clEnumVal(pbbs2vgr, "Convert pbbs graph to binary void gr"),
      clEnumVal(rmat2gr, "Convert rmat to binary gr"),
      clEnumVal(vgr2bsml, "Convert binary void gr to binary sparse MATLAB matrix"),
      clEnumVal(vgr2cvgr, "Clean up binary void gr: remove self edges and multi-edges"),
      clEnumVal(vgr2edgelist, "Convert binary void gr to edgelist"),
      clEnumVal(vgr2svgr, "Convert binary void gr to symmetric graph by adding reverse edges"),
      clEnumValEnd), cll::init(edgelist2gr));

static void printStatus(size_t in_nodes, size_t in_edges, size_t out_nodes, size_t out_edges) {
  std::cout << "InGraph : |V| = " << in_nodes << ", |E| = " << in_edges << "\n";
  std::cout << "OutGraph: |V| = " << out_nodes << ", |E| = " << out_edges << "\n";
}

static void printStatus(size_t in_nodes, size_t in_edges) {
  printStatus(in_nodes, in_edges, in_nodes, in_edges);
}

/**
 * Just a bunch of pairs or triples:
 * src dst weight?
 */
template<typename EdgeTy>
void convert_edgelist2gr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraphParser Parser;
  typedef Galois::LargeArray<EdgeTy,true> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  Parser p;
  EdgeData edgeData;
  std::ifstream infile(infilename.c_str());

  size_t numNodes = 0;
  size_t numEdges = 0;

  while (infile) {
    size_t src;
    size_t dst;
    edge_value_type data;

    infile >> src >> dst;

    if (EdgeData::has_value)
      infile >> data;

    if (infile) {
      ++numEdges;
      if (src > numNodes)
        numNodes = src;
      if (dst > numNodes)
        numNodes = dst;
    }
  }

  numNodes++;
  p.setNumNodes(numNodes);
  p.setNumEdges(numEdges);
  p.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0); 
  edgeData.allocate(numEdges);

  infile.clear();
  infile.seekg(0, std::ios::beg);
  p.phase1();
  while (infile) {
    size_t src;
    size_t dst;
    edge_value_type data;

    infile >> src >> dst;

    if (EdgeData::has_value)
      infile >> data;

    if (infile) {
      p.incrementDegree(src);
    }
  }

  infile.clear();
  infile.seekg(0, std::ios::beg);
  p.phase2();
  while (infile) {
    size_t src;
    size_t dst;
    edge_value_type data;

    infile >> src >> dst;

    if (EdgeData::has_value)
      infile >> data;
    
    if (infile) {
      edgeData.set(p.addNeighbor(src, dst), data);
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);

  p.structureToFile(outfilename);
  printStatus(numNodes, numEdges);
}

template<typename EdgeTy>
void convert_gr2edgelist(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  typedef Galois::LargeArray<EdgeTy,true> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  Graph graph;
  graph.structureFromFile(infilename);

  std::ofstream file(outfilename.c_str());
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (EdgeData::has_value) {
        file << src << " " << dst << " " << graph.getEdgeData<edge_value_type>(jj) << "\n";
      } else {
        file << src << " " << dst << "\n";
      }
    }
  }
  file.close();

  printStatus(graph.size(), graph.sizeEdges());
}

template<typename EdgeTy>
void convert_gr2sgr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraph Graph;

  Graph ingraph;
  Graph outgraph;
  ingraph.structureFromFile(infilename);
  Galois::Graph::makeSymmetric<EdgeTy>(ingraph, outgraph);

  outgraph.structureToFile(outfilename);
  printStatus(ingraph.size(), ingraph.sizeEdges(), outgraph.size(), outgraph.sizeEdges());
}

template<typename EdgeTy>
struct IdLess {
  bool operator()(const Galois::Graph::EdgeSortValue<EdgeTy>& e1, const Galois::Graph::EdgeSortValue<EdgeTy>& e2) const {
    return e1.dst < e2.dst;
  }
};

/**
 * Removes self and multi-edges from a graph.
 */
template<typename EdgeTy>
void convert_gr2cgr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  
  Graph orig, graph;
  {
    // Original FileGraph is immutable because it is backed by a file
    orig.structureFromFile(infilename);
    graph.cloneFrom(orig);
  }

  size_t numEdges = 0;

  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    graph.sortEdges<EdgeTy>(src, IdLess<EdgeTy>());

    Graph::edge_iterator prev = graph.edge_end(src);
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (src == dst) {
      } else if (prev != ej && graph.getEdgeDst(prev) == dst) {
      } else {
        numEdges += 1;
      }
      prev = jj;
    }
  }

  if (numEdges == graph.sizeEdges()) {
    std::cout << "Graph already simplified; copy input to output\n";
    printStatus(graph.size(), graph.sizeEdges());
    graph.structureToFile(outfilename);
    return;
  }

  typedef Galois::Graph::FileGraphParser Parser;
  typedef Galois::LargeArray<EdgeTy,true> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Parser p;
  EdgeData edgeData;

  p.setNumNodes(graph.size());
  p.setNumEdges(numEdges);
  p.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0);
  edgeData.allocate(numEdges);

  p.phase1();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    Graph::edge_iterator prev = graph.edge_end(src);
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (src == dst) {
      } else if (prev != ej && graph.getEdgeDst(prev) == dst) {
      } else {
        p.incrementDegree(src);
      }
      prev = jj;
    }
  }

  p.phase2();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    Graph::edge_iterator prev = graph.edge_end(src);
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (src == dst) {
      } else if (prev != ej && graph.getEdgeDst(prev) == dst) {
      } else if (EdgeData::has_value) {
        edgeData.set(p.addNeighbor(src, dst), graph.getEdgeData<edge_value_type>(jj));
      } else {
        p.addNeighbor(src, dst);
      }
      prev = jj;
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);
  
  p.structureToFile(outfilename);
  printStatus(graph.size(), graph.sizeEdges(), p.size(), p.sizeEdges());
}

#if 1
void convert_rmat2gr(const std::string& infilename, const std::string& outfilename) { abort(); }
#else
void convert_rmat2gr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FirstGraph<uint32_t,int32_t,true> Graph;
  typedef Graph::GraphNode GNode;
  Graph graph;

  std::ifstream infile(infilename.c_str());

  // Skip to first non-comment line
  while (infile) {
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
    graph.addNode(n);
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

  outputGraph(outfilename, graph);
  printStatus(nnodes, edges_added, graph.size(), graph.sizeEdges());
}
#endif

// Example:
//  c Some file
//  c Comments
//  p XXX XXX <num nodes> <num edges>
//  a <src id> <dst id> <weight>
//  ....
void convert_dimacs2gr(const std::string& infilename, const std::string& outfilename) { 
  typedef Galois::Graph::FileGraphParser Parser;
  typedef Galois::LargeArray<int32_t,true> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  Parser p;
  EdgeData edgeData;
  uint32_t nnodes;
  size_t nedges;

  for (int phase = 0; phase < 2; ++phase) {
    std::ifstream infile(infilename.c_str());

    while (infile) {
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
    infile >> tmp >> tmp >> nnodes >> nedges;
    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    if (phase == 0) {
      p.setNumNodes(nnodes);
      p.setNumEdges(nedges);
      p.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0); 
      edgeData.allocate(nedges);
      p.phase1();
    } else {
      p.phase2();
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

      // 1 indexed
      if (phase == 0) {
        p.incrementDegree(cur_id - 1);
      } else {
        edgeData.set(p.addNeighbor(cur_id - 1, neighbor_id - 1), weight);
      }

      infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    infile.peek();
    if (!infile.eof()) {
      std::cerr << "Error: additional lines in file\n";
      abort();
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);

  p.structureToFile(outfilename);
  printStatus(p.size(), p.sizeEdges());
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
  typedef Galois::Graph::FileGraphParser Parser;
  typedef Galois::LargeArray<EdgeTy,true> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Parser p;

  std::ifstream infile(infilename.c_str());
  std::string header;
  uint32_t nnodes;
  size_t nedges;

  infile >> header >> nnodes >> nedges;
  if (header != "AdjacencyGraph") {
    std::cerr << "Error: unknown file format\n";
    abort();
  }

  p.setNumNodes(nnodes);
  p.setNumEdges(nedges);
  p.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0);

  size_t* offsets = new size_t[nnodes];
  for (size_t i = 0; i < nnodes; ++i) {
    infile >> offsets[i];
  }

  size_t* edges = new size_t[nedges];
  for (size_t i = 0; i < nedges; ++i) {
    infile >> edges[i];
  }

  p.phase1();
  for (uint32_t i = 0; i < nnodes; ++i) {
    size_t begin = offsets[i];
    size_t end = (i == nnodes - 1) ? nedges : offsets[i+1];
    p.incrementDegree(i, end - begin);
  }

  p.phase2();
  for (uint32_t i = 0; i < nnodes; ++i) {
    size_t begin = offsets[i];
    size_t end = (i == nnodes - 1) ? nedges : offsets[i+1];
    for (size_t j = begin; j < end; ++j) {
      size_t dst = edges[j];
      p.addNeighbor(i, dst);
    }
  }

  p.finish<void>();

  p.structureToFile(outfilename);
  printStatus(p.size(), p.sizeEdges());
}

template<typename EdgeTy>
void convert_gr2wpbbs(const std::string& infilename, const std::string& outfilename) {
  // Use FileGraph because it is basically in CSR format needed for pbbs
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  std::ofstream file(outfilename.c_str());
  file << "WeightedEdgeArray\n";
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      EdgeTy& weight = graph.getEdgeData<EdgeTy>(jj);
      file << src << " " << dst << " " << weight << "\n";
    }
  }
  file.close();

  printStatus(graph.size(), graph.sizeEdges());
}

void convert_gr2pbbs(const std::string& infilename, const std::string& outfilename) {
  // Use FileGraph because it is basically in CSR format needed for pbbs
  typedef Galois::Graph::FileGraph Graph;

  Graph graph;
  graph.structureFromFile(infilename);

  std::ofstream file(outfilename.c_str());
  file << "AdjacencyGraph\n" << graph.size() << "\n" << graph.sizeEdges() << "\n";
  // edgeid[i] is the end of i in FileGraph while it is the beginning of i in pbbs graph
  size_t last = std::distance(graph.edge_id_begin(), graph.edge_id_end());
  size_t count = 0;
  file << "0\n";
  for (Graph::edge_id_iterator ii = graph.edge_id_begin(), ei = graph.edge_id_end();
      ii != ei; ++ii, ++count) {
    if (count < last - 1)
      file << *ii << "\n";
  }
  for (Graph::node_id_iterator ii = graph.node_id_begin(), ei = graph.node_id_end(); ii != ei; ++ii) {
    file << *ii << "\n";
  }
  file.close();

  printStatus(graph.size(), graph.sizeEdges());
}

template<typename EdgeTy>
void convert_gr2dimacs(const std::string& infilename, const std::string& outfilename) { 
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  std::ofstream file(outfilename.c_str());
  file << "p sp " << graph.size() << " " << graph.sizeEdges() << "\n";
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      EdgeTy& weight = graph.getEdgeData<EdgeTy>(jj);
      file << "a " << src + 1 << " " << dst + 1 << " " << weight << "\n";
    }
  }
  file.close();

  printStatus(graph.size(), graph.sizeEdges());
}

template<typename EdgeTy>
struct GetEdgeData {
  double operator()(Galois::Graph::FileGraph& g, Galois::Graph::FileGraph::edge_iterator ii) const {
    return g.getEdgeData<EdgeTy>(ii);
  }
};

template<>
struct GetEdgeData<void> {
  double operator()(Galois::Graph::FileGraph& g, Galois::Graph::FileGraph::edge_iterator ii) const {
    return 1;
  }
};

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
  typedef Galois::Graph::FileGraph Graph;
  typedef typename Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  uint32_t nnodes = graph.size();
  uint32_t nedges = graph.sizeEdges(); 

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
    uint32_t sid = src;
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
      uint32_t did = dst;
      retval = write(fd, &did, sizeof(did));
      if (retval == -1) { perror(__FILE__); abort(); }
    }
  }

  // Write data
  GetEdgeData<EdgeTy> convert;
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (typename Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      double weight = convert(graph, jj);
      retval = write(fd, &weight, sizeof(weight));
      if (retval == -1) { perror(__FILE__); abort(); }
    }
  }

  close(fd);
  printStatus(nnodes, nedges);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  switch (convertMode) {
    case dimacs2gr: convert_dimacs2gr(inputfilename, outputfilename); break;
    case edgelist2gr: convert_edgelist2gr<void>(inputfilename, outputfilename); break;
    case floatedgelist2gr: convert_edgelist2gr<float>(inputfilename, outputfilename); break;
    case gr2bsml: convert_gr2bsml<int32_t>(inputfilename, outputfilename); break;
    case gr2dimacs: convert_gr2dimacs<int32_t>(inputfilename, outputfilename); break;
    case gr2pbbs: convert_gr2pbbs(inputfilename, outputfilename); break;
    case gr2floatpbbs: convert_gr2wpbbs<float>(inputfilename, outputfilename); break;
    case gr2intpbbs: convert_gr2wpbbs<int32_t>(inputfilename, outputfilename); break;
    case intedgelist2gr: convert_edgelist2gr<int>(inputfilename, outputfilename); break;
    case pbbs2gr: convert_pbbs2gr<int32_t>(inputfilename, outputfilename); break;
    case pbbs2vgr: convert_pbbs2gr<void>(inputfilename, outputfilename); break;
    case rmat2gr: convert_rmat2gr(inputfilename, outputfilename); break;
    case vgr2bsml: convert_gr2bsml<void>(inputfilename, outputfilename); break;
    case vgr2edgelist: convert_gr2edgelist<void>(inputfilename, outputfilename); break;
    case vgr2svgr: convert_gr2sgr<void>(inputfilename, outputfilename); break;
    case vgr2cvgr: convert_gr2cgr<void>(inputfilename, outputfilename); break;
    default: abort();
  }
  return 0;
}
