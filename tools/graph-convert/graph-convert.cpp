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
#include "Galois/config.h"
#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <fstream>
#include <vector>
#include GALOIS_CXX11_STD_HEADER(random)

#include <fcntl.h>
#include <cstdlib>

namespace cll = llvm::cl;

enum ConvertMode {
  dimacs2gr,
  edgelist2vgr,
  floatedgelist2gr,
  gr2bsml,
  gr2cintgr,
  gr2dimacs,
  gr2doublemtx,
  gr2floatmtx,
  gr2floatpbbsedges,
  gr2intpbbs,
  gr2intpbbsedges,
  gr2lowdegreeintgr,
  gr2partdstintgr,
  gr2partsrcintgr,
  gr2randintgr,
  gr2sorteddstintgr,
  gr2sortedweightintgr,
  gr2ringintgr,
  gr2rmat,
  gr2sintgr,
  gr2tintgr,
  gr2treeintgr,
  intedgelist2gr,
  mtx2doublegr,
  mtx2floatgr,
  nodelist2vgr,
  pbbs2vgr,
  vgr2bsml,
  vgr2cvgr,
  vgr2edgelist,
  vgr2intgr,
  vgr2lowdegreevgr,
  vgr2pbbs,
  vgr2ringvgr,
  vgr2svgr,
  vgr2treevgr,
  vgr2trivgr,
  vgr2tvgr,
  vgr2vbinpbbs32,
  vgr2vbinpbbs64
};

static cll::opt<std::string> inputfilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputfilename(cll::Positional, cll::desc("<output file>"), cll::Required);
static cll::opt<ConvertMode> convertMode(cll::desc("Choose a conversion mode:"),
    cll::values(
      clEnumVal(dimacs2gr, "Convert dimacs to binary gr"),
      clEnumVal(edgelist2vgr, "Convert edge list to binary void gr"),
      clEnumVal(floatedgelist2gr, "Convert weighted (float) edge list to binary gr"),
      clEnumVal(gr2bsml, "Convert binary gr to binary sparse MATLAB matrix"),
      clEnumVal(gr2cintgr, "Clean up binary weighted (int) gr: remove self edges and multi-edges"),
      clEnumVal(gr2dimacs, "Convert binary gr to dimacs"),
      clEnumVal(gr2doublemtx, "Convert binary gr to matrix market format"),
      clEnumVal(gr2floatmtx, "Convert binary gr to matrix market format"),
      clEnumVal(gr2floatpbbsedges, "Convert binary gr to weighted (float) pbbs edge list"),
      clEnumVal(gr2intpbbs, "Convert binary gr to weighted (int) pbbs graph"),
      clEnumVal(gr2intpbbsedges, "Convert binary gr to weighted (int) pbbs edge list"),
      clEnumVal(gr2lowdegreeintgr, "Remove high degree nodes from binary gr"),
      clEnumVal(gr2partdstintgr, "Partition binary weighted (int) gr by destination nodes into N pieces"),
      clEnumVal(gr2partsrcintgr, "Partition binary weighted (int) gr by source nodes into N pieces"),
      clEnumVal(gr2randintgr, "Randomize binary weighted (int) gr"),
      clEnumVal(gr2ringintgr, "Convert binary gr to strongly connected graph by adding ring overlay"),
      clEnumVal(gr2rmat, "Convert binary gr to RMAT graph"),
      clEnumVal(gr2sintgr, "Convert binary gr to symmetric graph by adding reverse edges"),
      clEnumVal(gr2sorteddstintgr, "Sort outgoing edges of binary weighted (int) gr by edge destination"),
      clEnumVal(gr2sortedweightintgr, "Sort outgoing edges of binary weighted (int) gr by edge weight"),
      clEnumVal(gr2tintgr, "Transpose binary weighted (int) gr"),
      clEnumVal(gr2treeintgr, "Convert binary gr to strongly connected graph by adding tree overlay"),
      clEnumVal(intedgelist2gr, "Convert weighted (int) edge list to binary gr"),
      clEnumVal(mtx2doublegr, "Convert matrix market format to binary gr"),
      clEnumVal(mtx2floatgr, "Convert matrix market format to binary gr"),
      clEnumVal(nodelist2vgr, "Convert node list to binary gr"),
      clEnumVal(pbbs2vgr, "Convert pbbs graph to binary void gr"),
      clEnumVal(vgr2bsml, "Convert binary void gr to binary sparse MATLAB matrix"),
      clEnumVal(vgr2cvgr, "Clean up binary void gr: remove self edges and multi-edges"),
      clEnumVal(vgr2edgelist, "Convert binary void gr to edgelist"),
      clEnumVal(vgr2intgr, "Convert void binary gr to weighted (int) gr by adding random edge weights"),
      clEnumVal(vgr2lowdegreevgr, "Remove high degree nodes from binary gr"),
      clEnumVal(vgr2pbbs, "Convert binary gr to unweighted pbbs graph"),
      clEnumVal(vgr2ringvgr, "Convert binary gr to strongly connected graph by adding ring overlay"),
      clEnumVal(vgr2svgr, "Convert binary void gr to symmetric graph by adding reverse edges"),
      clEnumVal(vgr2treevgr, "Convert binary gr to strongly connected graph by adding tree overlay"),
      clEnumVal(vgr2trivgr, "Convert symmetric binary void gr to triangular form by removing reverse edges"),
      clEnumVal(vgr2tvgr, "Transpose binary gr"),
      clEnumVal(vgr2vbinpbbs32, "Convert binary gr to unweighted binary pbbs graph"),
      clEnumVal(vgr2vbinpbbs64, "Convert binary gr to unweighted binary pbbs graph"),
      clEnumValEnd), cll::Required);
static cll::opt<int> numParts("numParts", 
    cll::desc("number of parts to partition graph into"), cll::init(64));
static cll::opt<int> maxValue("maxValue",
    cll::desc("maximum weight to add (tree/ring edges are maxValue + 1)"), cll::init(100));
static cll::opt<int> maxDegree("maxDegree",
    cll::desc("maximum degree to keep"), cll::init(2*1024));

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
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  Writer p;
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
  p.setSizeofEdgeData(EdgeData::sizeof_value);
  edgeData.create(numEdges);

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

/**
 * Convert matrix market matrix to binary graph.
 *
 * %% comments
 * % ...
 * <num nodes> <num nodes> <num edges>
 * <src> <dst> <float>
 *
 * src and dst start at 1.
 */
template<typename EdgeTy>
void convert_mtx2gr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  Writer p;
  EdgeData edgeData;
  uint32_t nnodes;
  size_t nedges;

  for (int phase = 0; phase < 2; ++phase) {
    std::ifstream infile(infilename.c_str());

    // Skip comments
    while (infile) {
      if (infile.peek() != '%') {
        break;
      }
      infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    // Read header
    char header[256];
    infile.getline(header, 256, '\n');
    std::istringstream line(header, std::istringstream::in);
    std::vector<std::string> tokens;
    while (line) {
      std::string tmp;
      line >> tmp;
      if (line) {
        tokens.push_back(tmp);
      }
    }
    if (tokens.size() != 3) {
      GALOIS_DIE("Unknown problem specification line: ", line.str());
    }
    // Prefer C functions for maximum compatibility
    //nnodes = std::stoull(tokens[0]);
    //nedges = std::stoull(tokens[2]);
    nnodes = strtoull(tokens[0].c_str(), NULL, 0);
    nedges = strtoull(tokens[2].c_str(), NULL, 0);

    // Parse edges
    if (phase == 0) {
      p.setNumNodes(nnodes);
      p.setNumEdges(nedges);
      p.setSizeofEdgeData(EdgeData::sizeof_value);
      edgeData.create(nedges);
      p.phase1();
    } else {
      p.phase2();
    }

    for (size_t edge_num = 0; edge_num < nedges; ++edge_num) {
      uint32_t cur_id, neighbor_id;
      edge_value_type weight = 1;

      infile >> cur_id >> neighbor_id >> weight;
      if (cur_id == 0 || cur_id > nnodes) {
        GALOIS_DIE("Error: node id out of range: ", cur_id);
      }
      if (neighbor_id == 0 || neighbor_id > nnodes) {
        GALOIS_DIE("Error: neighbor id out of range: ", neighbor_id);
      }

      // 1 indexed
      if (phase == 0) {
        p.incrementDegree(cur_id - 1);
        //p.incrementDegree(neighbor_id - 1);
      } else {
        edgeData.set(p.addNeighbor(cur_id - 1, neighbor_id - 1), weight);
        //edgeData.set(p.addNeighbor(neighbor_id - 1, cur_id - 1), weight);
      }

      infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    infile.peek();
    if (!infile.eof()) {
      GALOIS_DIE("Error: additional lines in file");
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);

  p.structureToFile(outfilename);
  printStatus(p.size(), p.sizeEdges());
}

template<typename EdgeTy>
void convert_gr2mtx(const std::string& infilename, const std::string& outfilename) { 
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  std::ofstream file(outfilename.c_str());
  file << graph.size() << " " << graph.size() << " " << graph.sizeEdges() << "\n";
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      EdgeTy& weight = graph.getEdgeData<EdgeTy>(jj);
      file << src + 1 << " " << dst + 1 << " " << weight << "\n";
    }
  }
  file.close();

  printStatus(graph.size(), graph.sizeEdges());
}

/**
 * List of node adjacencies:
 *
 * <node id> <num neighbors> <neighbor id>*
 * ...
 */
void convert_nodelist2vgr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraphWriter Writer;

  Writer p;
  std::ifstream infile(infilename.c_str());

  size_t numNodes = 0;
  size_t numEdges = 0;

  while (infile) {
    size_t src;
    size_t numNeighbors;

    infile >> src >> numNeighbors;

    if (infile) {
      if (src > numNodes)
        numNodes = src;
      numEdges += numNeighbors;
    }
    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  numNodes++;
  p.setNumNodes(numNodes);
  p.setNumEdges(numEdges);

  infile.clear();
  infile.seekg(0, std::ios::beg);
  p.phase1();
  while (infile) {
    size_t src;
    size_t numNeighbors;

    infile >> src >> numNeighbors;

    if (infile) {
      p.incrementDegree(src, numNeighbors);
    }
    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  infile.clear();
  infile.seekg(0, std::ios::beg);
  p.phase2();
  while (infile) {
    size_t src;
    size_t numNeighbors;

    infile >> src >> numNeighbors;
    
    for (; infile && numNeighbors > 0; --numNeighbors) {
      size_t dst;
      infile >> dst;
      if (infile)
        p.addNeighbor(src, dst);
    }

    infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  p.finish<void>();

  p.structureToFile(outfilename);
  printStatus(numNodes, numEdges);
}


template<typename EdgeTy>
void convert_gr2edgelist(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
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

//! Wrap generator into form form std::random_shuffle
template<typename T,typename Gen,template<typename> class Dist>
struct UniformDist {
  Gen& gen;
  
  UniformDist(Gen& g): gen(g) { }
  T operator()(T m) {
    Dist<T> r(0, m - 1);
    return r(gen);
  }
};

template<typename EdgeTy>
void convert_gr2rand(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  typedef Galois::LargeArray<GNode> Permutation;
  typedef typename std::iterator_traits<typename Permutation::iterator>::difference_type difference_type;

  Graph graph;
  graph.structureFromFile(infilename);

  Permutation perm;
  perm.create(graph.size());
  std::copy(boost::counting_iterator<GNode>(0), boost::counting_iterator<GNode>(graph.size()), perm.begin());
  std::mt19937 gen;
#if __cplusplus >= 201103L || defined(HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
  UniformDist<difference_type,std::mt19937,std::uniform_int_distribution> dist(gen);
#else
  UniformDist<difference_type,std::mt19937,std::uniform_int> dist(gen);
#endif
  std::random_shuffle(perm.begin(), perm.end(), dist);

  Graph out;
  Galois::Graph::permute<EdgeTy>(graph, perm, out);

  out.structureToFile(outfilename);
  printStatus(graph.size(), graph.sizeEdges());
}

template<typename InEdgeTy,typename OutEdgeTy>
void add_weights(const std::string& infilename, const std::string& outfilename, int maxvalue) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  
  Graph graph, outgraph;

  graph.structureFromFile(infilename);
  OutEdgeTy* edgeData = outgraph.structureFromGraph<OutEdgeTy>(graph);
  OutEdgeTy* edgeDataEnd = edgeData + graph.sizeEdges();

  std::mt19937 gen;
#if __cplusplus >= 201103L || defined(HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
  std::uniform_int_distribution<OutEdgeTy> dist(1, maxvalue);
#else
  std::uniform_int<OutEdgeTy> dist(1, maxvalue);
#endif
  for (; edgeData != edgeDataEnd; ++edgeData) {
    *edgeData = dist(gen);
  }
  
  outgraph.structureToFile(outfilename);
  printStatus(graph.size(), graph.sizeEdges(), outgraph.size(), outgraph.sizeEdges());
}

template<typename EdgeValue,bool Enable>
void setEdgeValue(EdgeValue& edgeValue, int maxvalue, typename std::enable_if<Enable>::type* = 0) {
  edgeValue.set(0, maxvalue + 1);
}

template<typename EdgeValue,bool Enable>
void setEdgeValue(EdgeValue& edgeValue, int maxvalue, typename std::enable_if<!Enable>::type* = 0) { }

/**
 * Add edges (i, i-1) for all i \in V.
 */
template<typename EdgeTy>
void add_ring(const std::string& infilename, const std::string& outfilename, int maxvalue) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Graph::GraphNode GNode;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Graph graph;
  graph.structureFromFile(infilename);

  Writer p;
  EdgeData edgeData;
  EdgeData edgeValue;

  uint64_t size = graph.size();

  p.setNumNodes(size);
  p.setNumEdges(graph.sizeEdges() + size);
  p.setSizeofEdgeData(EdgeData::sizeof_value);
  edgeData.create(graph.sizeEdges() + size);
  edgeValue.create(1);
  //edgeValue.set(0, maxValue + 1);
  setEdgeValue<EdgeData,EdgeData::has_value>(edgeValue, maxvalue);

  p.phase1();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    p.incrementDegree(src, std::distance(graph.edge_begin(src), graph.edge_end(src)) + 1);
  }

  p.phase2();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (EdgeData::has_value) {
        edgeData.set(p.addNeighbor(src, dst), graph.getEdgeData<edge_value_type>(jj));
      } else {
        p.addNeighbor(src, dst);
      }
    }

    GNode dst = src == 0 ? size - 1 : src - 1;
    if (EdgeData::has_value) {
      edgeData.set(p.addNeighbor(src, dst), edgeValue.at(0));
    } else {
      p.addNeighbor(src, dst);
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);
  p.structureToFile(outfilename);
  printStatus(graph.size(), graph.sizeEdges(), p.size(), p.sizeEdges());
}

/**
 * Add edges (i, i*2+1), (i, i*2+2) and their complement. 
 */
template<typename EdgeTy>
void add_tree(const std::string& infilename, const std::string& outfilename, int maxvalue) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Graph::GraphNode GNode;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Graph graph;
  graph.structureFromFile(infilename);

  Writer p;
  EdgeData edgeData;
  EdgeData edgeValue;

  uint64_t size = graph.size();
  uint64_t newEdges = 0;
  if (size >= 2) {
    // Closed form counts for the loop below 
    newEdges =  (size - 1 + (2 - 1)) / 2; // (1) rounded up
    newEdges += (size - 2 + (2 - 1)) / 2; // (2) rounded up
  } else if (size >= 1)
    newEdges = 1;
  newEdges *= 2; // reverse edges

  p.setNumNodes(size);
  p.setNumEdges(graph.sizeEdges() + newEdges);
  p.setSizeofEdgeData(EdgeData::sizeof_value);
  edgeData.create(graph.sizeEdges() + newEdges);
  edgeValue.create(1);
  //edgeValue.set(0, maxValue + 1);
  setEdgeValue<EdgeData,EdgeData::has_value>(edgeValue, maxvalue);

  p.phase1();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    p.incrementDegree(src, std::distance(graph.edge_begin(src), graph.edge_end(src)));
    if (src * 2 + 1 < size) { // (1)
      p.incrementDegree(src);
      p.incrementDegree(src * 2 + 1);
    }
    if (src * 2 + 2 < size) { // (2)
      p.incrementDegree(src);
      p.incrementDegree(src * 2 + 2);
    }
  }

  p.phase2();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (EdgeData::has_value) {
        edgeData.set(p.addNeighbor(src, dst), graph.getEdgeData<edge_value_type>(jj));
      } else {
        p.addNeighbor(src, dst);
      }
    }
    if (src * 2 + 1 < size) {
      if (EdgeData::has_value) {
        edgeData.set(p.addNeighbor(src, src * 2 + 1), edgeValue.at(0));
        edgeData.set(p.addNeighbor(src * 2 + 1, src), edgeValue.at(0));
      } else {
        p.addNeighbor(src, src * 2 + 1);
        p.addNeighbor(src * 2 + 1, src);
      }
    }
    if (src * 2 + 2 < size) {
      if (EdgeData::has_value) {
        edgeData.set(p.addNeighbor(src, src * 2 + 2), edgeValue.at(0));
        edgeData.set(p.addNeighbor(src * 2 + 2, src), edgeValue.at(0));
      } else {
        p.addNeighbor(src, src * 2 + 2);
        p.addNeighbor(src * 2 + 2, src);
      }
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);
  p.structureToFile(outfilename);
  printStatus(graph.size(), graph.sizeEdges(), p.size(), p.sizeEdges());
}

//! Make graph symmetric by blindly adding reverse entries
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
void remove_high_degree(const std::string& infilename, const std::string& outfilename, int degree) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Graph graph;
  graph.structureFromFile(infilename);

  Writer p;
  EdgeData edgeData;

  std::vector<GNode> nodeTable;
  nodeTable.resize(graph.size());
  uint64_t numNodes = 0;
  uint64_t numEdges = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src);
    if (std::distance(jj, ej) > degree)
      continue;
    nodeTable[src] = numNodes++;
    for (; jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (std::distance(graph.edge_begin(dst), graph.edge_end(dst)) > degree)
        continue;
      ++numEdges;
    }
  }

  if (numEdges == graph.sizeEdges() && numNodes == graph.size()) {
    std::cout << "Graph already simplified; copy input to output\n";
    printStatus(graph.size(), graph.sizeEdges());
    graph.structureToFile(outfilename);
    return;
  }

  p.setNumNodes(numNodes);
  p.setNumEdges(numEdges);
  p.setSizeofEdgeData(EdgeData::sizeof_value);
  edgeData.create(numEdges);

  p.phase1();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src);
    if (std::distance(jj, ej) > degree)
      continue;
    for (; jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (std::distance(graph.edge_begin(dst), graph.edge_end(dst)) > degree)
        continue;
      p.incrementDegree(nodeTable[src]);
    }
  }

  p.phase2();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src);
    if (std::distance(jj, ej) > degree)
      continue;
    for (; jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (std::distance(graph.edge_begin(dst), graph.edge_end(dst)) > degree)
        continue;
      if (EdgeData::has_value) {
        edgeData.set(p.addNeighbor(nodeTable[src], nodeTable[dst]), graph.getEdgeData<edge_value_type>(jj));
      } else {
        p.addNeighbor(nodeTable[src], nodeTable[dst]);
      }
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);
  
  p.structureToFile(outfilename);
  printStatus(graph.size(), graph.sizeEdges(), p.size(), p.sizeEdges());
}

//! Partition graph into balanced number of edges by source node
template<typename EdgeTy>
void partition_by_source(const std::string& infilename, const std::string& outfilename, int parts) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Graph graph;
  graph.structureFromFile(infilename);

  for (int i = 0; i < parts; ++i) {
    Writer p;
    EdgeData edgeData;

    auto r = graph.divideBy(0, 1, i, parts);

    size_t numEdges = 0;
    if (r.first != r.second)
      numEdges = std::distance(graph.edge_begin(*r.first), graph.edge_end(*(r.second - 1)));

    p.setNumNodes(graph.size());
    p.setNumEdges(numEdges);
    p.setSizeofEdgeData(EdgeData::sizeof_value);
    edgeData.create(numEdges);

    p.phase1();
    for (Graph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      GNode src = *ii;
      p.incrementDegree(src, std::distance(graph.edge_begin(src), graph.edge_end(src)));
    }
    
    p.phase2();
    for (Graph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      GNode src = *ii;
      for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        if (EdgeData::has_value)
          edgeData.set(p.addNeighbor(src, dst), graph.getEdgeData<edge_value_type>(jj));
        else
          p.addNeighbor(src, dst);
      }
    }

    edge_value_type* rawEdgeData = p.finish<edge_value_type>();
    if (EdgeData::has_value)
      std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);

    std::ostringstream partname;
    partname << outfilename << "." << i << ".of." << parts;

    p.structureToFile(partname.str());
    printStatus(graph.size(), graph.sizeEdges(), p.size(), p.sizeEdges());
  }
}

template<typename InDegree, typename It = typename InDegree::iterator>
static std::pair<It,It> divide_by_destination(InDegree& inDegree, int id, int total) 
{
  if (inDegree.begin() == inDegree.end())
    return std::make_pair(inDegree.begin(), inDegree.end());

  size_t size = inDegree[inDegree.size() - 1];
  size_t block = (size + total - 1) / total;

  It bb = std::lower_bound(inDegree.begin(), inDegree.end(), id * block);
  It eb;
  if (id + 1 == total)
    eb = inDegree.end();
  else 
    eb = std::upper_bound(bb, inDegree.end(), (id + 1) * block);
  return std::make_pair(bb, eb);
}

template<typename GraphTy, typename InDegree>
static void compute_indegree(GraphTy& graph, InDegree& inDegree) {
  inDegree.create(graph.size());

  for (auto nn = graph.begin(), en = graph.end(); nn != en; ++nn) {
    for (auto jj = graph.edge_begin(*nn), ej = graph.edge_end(*nn); jj != ej; ++jj) {
      auto dst = graph.getEdgeDst(jj);
      inDegree[dst] += 1;
    }
  }

  for (size_t i = 1; i < inDegree.size(); ++i)
    inDegree[i] = inDegree[i-1] + inDegree[i];
}

//! Partition graph into balanced number of edges by destination node
template<typename EdgeTy>
void partition_by_destination(const std::string& infilename, const std::string& outfilename, int parts) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef Galois::LargeArray<size_t> InDegree;
  typedef typename EdgeData::value_type edge_value_type;
  
  Graph graph;
  graph.structureFromFile(infilename);
  InDegree inDegree;
  compute_indegree(graph, inDegree);

  for (int i = 0; i < parts; ++i) {
    Writer p;
    EdgeData edgeData;

    auto r = divide_by_destination(inDegree, i, parts);
    size_t bb = std::distance(inDegree.begin(), r.first);
    size_t eb = std::distance(inDegree.begin(), r.second);

    size_t numEdges = 0;
    if (bb != eb) {
      size_t begin = bb == 0 ? 0 : inDegree[bb - 1];
      size_t end = eb == 0 ? 0 : inDegree[eb - 1];
      numEdges = end - begin;
    }

    p.setNumNodes(graph.size());
    p.setNumEdges(numEdges);
    p.setSizeofEdgeData(EdgeData::sizeof_value);
    edgeData.create(numEdges);

    p.phase1();
    for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      GNode src = *ii;
      for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        if (dst < bb)
          continue;
        if (dst >= eb)
          continue;
        p.incrementDegree(src);
      }
    }
    
    p.phase2();
    for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      GNode src = *ii;
      for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);
        if (dst < bb)
          continue;
        if (dst >= eb)
          continue;
        if (EdgeData::has_value)
          edgeData.set(p.addNeighbor(src, dst), graph.getEdgeData<edge_value_type>(jj));
        else
          p.addNeighbor(src, dst);
      }
    }

    edge_value_type* rawEdgeData = p.finish<edge_value_type>();
    if (EdgeData::has_value)
      std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);

    std::ostringstream partname;
    partname << outfilename << "." << i << ".of." << parts;

    p.structureToFile(partname.str());
    printStatus(graph.size(), graph.sizeEdges(), p.size(), p.sizeEdges());
  }
}

//! Transpose graph
template<typename EdgeTy>
void transpose(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Graph graph;
  graph.structureFromFile(infilename);

  Writer p;
  EdgeData edgeData;

  p.setNumNodes(graph.size());
  p.setNumEdges(graph.sizeEdges());
  p.setSizeofEdgeData(EdgeData::sizeof_value);
  edgeData.create(graph.sizeEdges());

  p.phase1();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      p.incrementDegree(dst);
    }
  }

  p.phase2();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (EdgeData::has_value) {
        edgeData.set(p.addNeighbor(dst, src), graph.getEdgeData<edge_value_type>(jj));
      } else {
        p.addNeighbor(dst, src);
      }
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);
  
  p.structureToFile(outfilename);
  printStatus(graph.size(), graph.sizeEdges(), p.size(), p.sizeEdges());
}

template<typename GraphNode,typename EdgeTy>
struct IdLess {
  bool operator()(const Galois::Graph::EdgeSortValue<GraphNode,EdgeTy>& e1, const Galois::Graph::EdgeSortValue<GraphNode,EdgeTy>& e2) const {
    return e1.dst < e2.dst;
  }
};

template<typename GraphNode,typename EdgeTy>
struct WeightLess {
  bool operator()(const Galois::Graph::EdgeSortValue<GraphNode,EdgeTy>& e1, const Galois::Graph::EdgeSortValue<GraphNode,EdgeTy>& e2) const {
    return e1.get() < e2.get();
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

    graph.sortEdges<EdgeTy>(src, IdLess<GNode,EdgeTy>());

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

  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Writer p;
  EdgeData edgeData;

  p.setNumNodes(graph.size());
  p.setNumEdges(numEdges);
  p.setSizeofEdgeData(EdgeData::sizeof_value);
  edgeData.create(numEdges);

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

template<typename EdgeTy,template<typename,typename> class SortBy>
void sort_edges(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Graph orig, graph;
  {
    // Original FileGraph is immutable because it is backed by a file
    orig.structureFromFile(infilename);
    graph.cloneFrom(orig);
  }

  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    graph.sortEdges<EdgeTy>(src, SortBy<GNode,EdgeTy>());
  }

  graph.structureToFile(outfilename);
  printStatus(graph.size(), graph.sizeEdges());
}

/**
 * Removes edges such that src > dst
 */
template<typename EdgeTy>
void convert_sgr2gr(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;
  
  Graph graph;
  graph.structureFromFile(infilename);

  size_t numEdges = 0;

  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (src > dst) {
      } else {
        numEdges += 1;
      }
    }
  }

  if (numEdges == graph.sizeEdges()) {
    std::cout << "Graph already simplified; copy input to output\n";
    printStatus(graph.size(), graph.sizeEdges());
    graph.structureToFile(outfilename);
    return;
  }

  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;
  
  Writer p;
  EdgeData edgeData;

  p.setNumNodes(graph.size());
  p.setNumEdges(numEdges);
  p.setSizeofEdgeData(EdgeData::sizeof_value);
  edgeData.create(numEdges);

  p.phase1();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (src > dst) {
      } else {
        p.incrementDegree(src);
      }
    }
  }

  p.phase2();
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;

    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      if (src > dst) {
      } else if (EdgeData::has_value) {
        edgeData.set(p.addNeighbor(src, dst), graph.getEdgeData<edge_value_type>(jj));
      } else {
        p.addNeighbor(src, dst);
      }
    }
  }

  edge_value_type* rawEdgeData = p.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);
  
  p.structureToFile(outfilename);
  printStatus(graph.size(), graph.sizeEdges(), p.size(), p.sizeEdges());
}

// Example:
//  c Some file
//  c Comments
//  p XXX* <num nodes> <num edges>
//  a <src id> <dst id> <weight>
//  ....
void convert_dimacs2gr(const std::string& infilename, const std::string& outfilename) { 
  typedef Galois::Graph::FileGraphWriter Writer;
  typedef Galois::LargeArray<int32_t> EdgeData;
  typedef EdgeData::value_type edge_value_type;

  Writer p;
  EdgeData edgeData;
  uint32_t nnodes;
  size_t nedges;

  for (int phase = 0; phase < 2; ++phase) {
    std::ifstream infile(infilename.c_str());

    // Skip comments
    while (infile) {
      if (infile.peek() == 'p') {
        break;
      }
      infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    // Read header
    char header[256];
    infile.getline(header, 256, '\n');
    std::istringstream line(header, std::istringstream::in);
    std::vector<std::string> tokens;
    while (line) {
      std::string tmp;
      line >> tmp;
      if (line) {
        tokens.push_back(tmp);
      }
    }
    if (tokens.size() < 3 || tokens[0].compare("p") != 0) {
      GALOIS_DIE("Unknown problem specification line: ", line.str());
    }
    // Prefer C functions for maximum compatibility
    //nnodes = std::stoull(tokens[tokens.size() - 2]);
    //nedges = std::stoull(tokens[tokens.size() - 1]);
    nnodes = strtoull(tokens[tokens.size() - 2].c_str(), NULL, 0);
    nedges = strtoull(tokens[tokens.size() - 1].c_str(), NULL, 0);

    // Parse edges
    if (phase == 0) {
      p.setNumNodes(nnodes);
      p.setNumEdges(nedges);
      p.setSizeofEdgeData(EdgeData::sizeof_value);
      edgeData.create(nedges);
      p.phase1();
    } else {
      p.phase2();
    }

    for (size_t edge_num = 0; edge_num < nedges; ++edge_num) {
      uint32_t cur_id, neighbor_id;
      int32_t weight;
      std::string tmp;
      infile >> tmp;

      if (tmp.compare("a") != 0) {
        --edge_num;
        infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        continue;
      }

      infile >> cur_id >> neighbor_id >> weight;
      if (cur_id == 0 || cur_id > nnodes) {
        GALOIS_DIE("Error: node id out of range: ", cur_id);
      }
      if (neighbor_id == 0 || neighbor_id > nnodes) {
        GALOIS_DIE("Error: neighbor id out of range: ", neighbor_id);
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
      GALOIS_DIE("Error: additional lines in file");
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
void convert_pbbs2vgr(const std::string& infilename, const std::string& outfilename) { 
  typedef Galois::Graph::FileGraphWriter Writer;
  
  Writer p;

  std::ifstream infile(infilename.c_str());
  std::string header;
  uint32_t nnodes;
  size_t nedges;

  infile >> header >> nnodes >> nedges;
  if (header != "AdjacencyGraph") {
    GALOIS_DIE("Error: unknown file format");
  }

  p.setNumNodes(nnodes);
  p.setNumEdges(nedges);

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
void convert_gr2pbbsedges(const std::string& infilename, const std::string& outfilename) {
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
      //file << src << "," << dst << "," << weight << "\n";
    }
  }
  file.close();

  printStatus(graph.size(), graph.sizeEdges());
}

/**
 * PBBS input is an ASCII file of tokens that serialize a CSR graph. I.e., 
 * elements in brackets are non-literals:
 * 
 * [Weighted]AdjacencyGraph
 * <num nodes>
 * <num edges>
 * <offset node 0>
 * <offset node 1>
 * ...
 * <edge 0>
 * <edge 1>
 * ...
 * [
 * <edge weight 0>
 * <edge weight 1>
 * ...
 * ]
 */
template<typename InEdgeTy,typename OutEdgeTy>
void convert_gr2pbbs(const std::string& infilename, const std::string& outfilename) {
  typedef Galois::Graph::FileGraph Graph;
  typedef Galois::LargeArray<OutEdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  Graph graph;
  graph.structureFromFile(infilename);

  std::ofstream file(outfilename.c_str());
  if (EdgeData::has_value)
    file << "Weighted";
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
  if (EdgeData::has_value) {
    for (edge_value_type* ii = graph.edge_data_begin<edge_value_type>(), *ei = graph.edge_data_end<edge_value_type>();
        ii != ei; ++ii) {
      file << *ii << "\n";
    }
  }
  file.close();

  printStatus(graph.size(), graph.sizeEdges());
}

/**
 * Binary PBBS format is three files.
 *
 * <base>.config - ASCII file with number of vertices
 * <base>.adj - Binary adjacencies
 * <base>.idx - Binary offsets for adjacencies
 */
template<typename NodeIdx,typename Offset>
void convert_gr2vbinpbbs(const std::string& infilename, const std::string& outfilename) { 
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  {
    std::string configName = outfilename + ".config";
    std::ofstream configFile(configName.c_str());
    configFile << graph.size() << "\n";
  }

  {
    std::string idxName = outfilename + ".idx";
    std::ofstream idxFile(idxName.c_str());
    // edgeid[i] is the end of i in FileGraph while it is the beginning of i in pbbs graph
    size_t last = std::distance(graph.edge_id_begin(), graph.edge_id_end());
    size_t count = 0;
    Offset offset = 0;
    idxFile.write(reinterpret_cast<char*>(&offset), sizeof(offset));
    for (Graph::edge_id_iterator ii = graph.edge_id_begin(), ei = graph.edge_id_end(); ii != ei; ++ii, ++count) {
      offset = *ii;
      if (count < last - 1)
        idxFile.write(reinterpret_cast<char*>(&offset), sizeof(offset));
    }
    idxFile.close();
  }

  {
    std::string adjName = outfilename + ".adj";
    std::ofstream adjFile(adjName.c_str());
    for (Graph::node_id_iterator ii = graph.node_id_begin(), ei = graph.node_id_end(); ii != ei; ++ii) {
      NodeIdx nodeIdx = *ii;
      adjFile.write(reinterpret_cast<char*>(&nodeIdx), sizeof(nodeIdx));
    }
    adjFile.close();
  }

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

/**
 * RMAT format (zero indexed):
 *  %%% Comment1
 *  %%% Comment2
 *  %%% Comment3
 *  <num nodes> <num edges>
 *  <node id> <num edges> [<neighbor id> <neighbor weight>]*
 *  ...
 */
template<typename InEdgeTy,typename OutEdgeTy>
void convert_gr2rmat(const std::string& infilename, const std::string& outfilename) { 
  typedef Galois::Graph::FileGraph Graph;
  typedef Graph::GraphNode GNode;

  Graph graph;
  graph.structureFromFile(infilename);

  std::ofstream file(outfilename.c_str());
  file << "%%%\n";
  file << "%%%\n";
  file << "%%%\n";
  file << graph.size() << " " << graph.sizeEdges() << "\n";
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    file << *ii << " " << std::distance(graph.edge_begin(src), graph.edge_end(src));
    for (Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      OutEdgeTy weight = graph.getEdgeData<InEdgeTy>(jj);
      file << " " << dst << " " << weight;
    }
    file << "\n";
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
  if (fd == -1) { GALOIS_SYS_DIE(""); }
  int retval;

  // Write header
  retval = write(fd, &nnodes, sizeof(nnodes));
  if (retval == -1) { GALOIS_SYS_DIE(""); }
  retval = write(fd, &nnodes, sizeof(nnodes));
  if (retval == -1) { GALOIS_SYS_DIE(""); }
  retval = write(fd, &nedges, sizeof(nedges));
  if (retval == -1) { GALOIS_SYS_DIE(""); }

  // Write row adjacency
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    uint32_t sid = src;
    for (typename Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      retval = write(fd, &sid, sizeof(sid));
      if (retval == -1) { GALOIS_SYS_DIE(""); }
    }
  }

  // Write column adjacency
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (typename Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      uint32_t did = dst;
      retval = write(fd, &did, sizeof(did));
      if (retval == -1) { GALOIS_SYS_DIE(""); }
    }
  }

  // Write data
  GetEdgeData<EdgeTy> convert;
  for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (typename Graph::edge_iterator jj = graph.edge_begin(src), ej = graph.edge_end(src); jj != ej; ++jj) {
      double weight = convert(graph, jj);
      retval = write(fd, &weight, sizeof(weight));
      if (retval == -1) { GALOIS_SYS_DIE(""); }
    }
  }

  close(fd);
  printStatus(nnodes, nedges);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  switch (convertMode) {
    case dimacs2gr: convert_dimacs2gr(inputfilename, outputfilename); break;
    case edgelist2vgr: convert_edgelist2gr<void>(inputfilename, outputfilename); break;
    case floatedgelist2gr: convert_edgelist2gr<float>(inputfilename, outputfilename); break;
    case gr2bsml: convert_gr2bsml<int32_t>(inputfilename, outputfilename); break;
    case gr2cintgr: convert_gr2cgr<int32_t>(inputfilename, outputfilename); break;
    case gr2dimacs: convert_gr2dimacs<int32_t>(inputfilename, outputfilename); break;
    case gr2doublemtx: convert_gr2mtx<double>(inputfilename, outputfilename); break;
    case gr2floatmtx: convert_gr2mtx<float>(inputfilename, outputfilename); break;
    case gr2floatpbbsedges: convert_gr2pbbsedges<float>(inputfilename, outputfilename); break;
#if !defined(__IBMCPP__) || __IBMCPP__ > 1210
    case gr2intpbbs: convert_gr2pbbs<int32_t,int32_t>(inputfilename, outputfilename); break;
#endif
    case gr2intpbbsedges: convert_gr2pbbsedges<int32_t>(inputfilename, outputfilename); break;
    case gr2lowdegreeintgr: remove_high_degree<int32_t>(inputfilename, outputfilename, maxDegree); break;
// XXX(ddn): The below triggers some internal XLC bug
#if !defined(__IBMCPP__) || __IBMCPP__ > 1210
    case gr2partdstintgr: partition_by_destination<int32_t>(inputfilename, outputfilename, numParts); break;
#endif
    case gr2partsrcintgr: partition_by_source<int32_t>(inputfilename, outputfilename, numParts); break;
    case gr2randintgr: convert_gr2rand<int32_t>(inputfilename, outputfilename); break;
    case gr2sorteddstintgr: sort_edges<int32_t,IdLess>(inputfilename, outputfilename); break;
    case gr2sortedweightintgr: sort_edges<int32_t,WeightLess>(inputfilename, outputfilename); break;
    case gr2ringintgr: add_ring<int32_t>(inputfilename, outputfilename, maxValue); break;
    case gr2rmat: convert_gr2rmat<int32_t,int32_t>(inputfilename, outputfilename); break;
    case gr2sintgr: convert_gr2sgr<int32_t>(inputfilename, outputfilename); break;
    case gr2tintgr: transpose<int32_t>(inputfilename, outputfilename); break;
    case gr2treeintgr: add_tree<int32_t>(inputfilename, outputfilename, maxValue); break;
    case intedgelist2gr: convert_edgelist2gr<int>(inputfilename, outputfilename); break;
    case mtx2doublegr: convert_mtx2gr<double>(inputfilename, outputfilename); break;
    case mtx2floatgr: convert_mtx2gr<float>(inputfilename, outputfilename); break;
    case nodelist2vgr: convert_nodelist2vgr(inputfilename, outputfilename); break;
    case pbbs2vgr: convert_pbbs2vgr(inputfilename, outputfilename); break;
    case vgr2bsml: convert_gr2bsml<void>(inputfilename, outputfilename); break;
    case vgr2cvgr: convert_gr2cgr<void>(inputfilename, outputfilename); break;
    case vgr2edgelist: convert_gr2edgelist<void>(inputfilename, outputfilename); break;
    case vgr2intgr: add_weights<void,int32_t>(inputfilename, outputfilename, maxValue); break;
    case vgr2lowdegreevgr: remove_high_degree<void>(inputfilename, outputfilename, maxDegree); break;
#if !defined(__IBMCPP__) || __IBMCPP__ > 1210
    case vgr2pbbs: convert_gr2pbbs<void,void>(inputfilename, outputfilename); break;
#endif
    case vgr2ringvgr: add_ring<void>(inputfilename, outputfilename, maxValue); break;
    case vgr2svgr: convert_gr2sgr<void>(inputfilename, outputfilename); break;
    case vgr2treevgr: add_tree<void>(inputfilename, outputfilename, maxValue); break;
    case vgr2trivgr: convert_sgr2gr<void>(inputfilename, outputfilename); break;
    case vgr2tvgr: transpose<void>(inputfilename, outputfilename); break;
#if !defined(__IBMCPP__) || __IBMCPP__ > 1210
    case vgr2vbinpbbs32: convert_gr2vbinpbbs<uint32_t,uint32_t>(inputfilename, outputfilename); break;
    case vgr2vbinpbbs64: convert_gr2vbinpbbs<uint32_t,uint64_t>(inputfilename, outputfilename); break;
#endif
    default: abort();
  }
  return 0;
}
