/** Graph converter -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
#include "Galois/Galois.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Runtime/OfflineGraph.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <vector>
#include <cstdlib>

namespace cll = llvm::cl;

enum StatMode {
  degreehist,
  degrees,
  maxDegreeNode,
  dsthist,
  indegreehist,
  sortedlogoffsethist,
  sparsityPattern,
  summary
};

static cll::opt<std::string> inputfilename(cll::Positional, cll::desc("<graph file>"), cll::Required);
static cll::list<StatMode> statModeList(cll::desc("Available stats:"),
    cll::values(
      clEnumVal(degreehist, "Histogram of degrees"),
      clEnumVal(degrees, "Node degrees"),
      clEnumVal(maxDegreeNode, "Max Degree Node"),
      clEnumVal(dsthist, "Histogram of destinations"),
      clEnumVal(indegreehist, "Histogram of indegrees"),
      clEnumVal(sortedlogoffsethist, "Histogram of neighbor offsets with sorted edges"),
      clEnumVal(sparsityPattern, "Pattern of non-zeros when graph is interpreted as a sparse matrix"),
      clEnumVal(summary, "Graph summary"),
      clEnumValEnd));
static cll::opt<int> numBins("numBins", cll::desc("Number of bins"), cll::init(-1));
static cll::opt<int> columns("columns", cll::desc("Columns for sparsity"), cll::init(80));

typedef galois::Graph::OfflineGraph Graph;
typedef Graph::GraphNode GNode;

void doSummary(Graph& graph) {
  std::cout << "NumNodes: " << graph.size() << "\n";
  std::cout << "NumEdges: " << graph.sizeEdges() << "\n";
  std::cout << "SizeofEdge: " << graph.edgeSize() << "\n";
}

void doDegrees(Graph& graph) {
  for (auto n : graph) {
    std::cout << std::distance(graph.edge_begin(n), graph.edge_end(n)) << "\n";
  }
}

void findMaxDegreeNode(Graph& graph){
  uint64_t nodeId = 0;
  size_t MaxDegree = 0;
  uint64_t MaxDegreeNode = 0;
  for (auto n : graph) {
    size_t degree = std::distance(graph.edge_begin(n), graph.edge_end(n));
    if(MaxDegree < degree){
      MaxDegree = degree;
      MaxDegreeNode = nodeId;
    }
    ++nodeId;
  }
  std::cout << "MaxDegreeNode : " << MaxDegreeNode << " , MaxDegree : " << MaxDegree << "\n";
}
void printHistogram(const std::string& name, std::map<uint64_t, uint64_t>& hists) {
  auto max = hists.rbegin()->first;
  if (numBins <= 0) {
    std::cout << name << "Bin,Start,End,Count\n";
    for (unsigned x = 0; x <= max; ++x) {
      std::cout << x << ',' << x << ',' << x+1 << ',';
      if (hists.count(x))
        std::cout << hists[x] << '\n';
      else
        std::cout << "0\n";
    }
  } else {
    std::vector<uint64_t> bins(numBins);    
    auto bwidth = (max+1) / numBins;
    if ((max+1) % numBins)
      ++bwidth;
    //    std::cerr << "* " << max << " " << numBins << " " << bwidth << "\n";
    for (auto p : hists)
      bins.at(p.first / bwidth) += p.second;
    std::cout << name << "Bin,Start,End,Count\n";
    for (unsigned x = 0; x < bins.size(); ++x)
      std::cout << x << ',' << x * bwidth << ',' << (x * bwidth + bwidth) << ',' << bins[x] << '\n';
  }
}


void doSparsityPattern(Graph& graph, 
                       std::function<void(unsigned, unsigned, bool)> printFn) {
  unsigned blockSize = (graph.size() + columns - 1) / columns;

  for (int i = 0; i < columns; ++i) {
    std::vector<bool> row(columns);
    auto p = galois::block_range(graph.begin(), graph.end(), i, columns);
    for (auto ii = p.first, ei = p.second; ii != ei; ++ii) {
      for (auto jj : graph.edges(*ii)) {
        row[graph.getEdgeDst(jj) / blockSize] = true;
      }
    }
    for (int x = 0; x < columns; ++x)
      printFn(x,i,row[x]);
  }
}

void doDegreeHistogram(Graph& graph) {
  std::map<uint64_t, uint64_t> hist;
  for (auto ii : graph)
    ++hist[std::distance(graph.edge_begin(ii), graph.edge_end(ii))];
  printHistogram("Degree", hist);
}

void doInDegreeHistogram(Graph& graph) {
  std::vector<uint64_t> inv(graph.size());
  std::map<uint64_t, uint64_t> hist;
  for (auto ii : graph)
    for (auto jj : graph.edges(ii))
      ++inv[graph.getEdgeDst(jj)];
  for (uint64_t n : inv)
    ++hist[n];
  printHistogram("InDegree", hist);
}

struct EdgeComp {
  typedef galois::Graph::EdgeSortValue<GNode, void> Edge;
  bool operator()(const Edge& a, const Edge& b) const {
    return a.dst < b.dst;
  }
};

int getLogIndex(ptrdiff_t x) {
  int logvalue = 0;
  int sign = x < 0 ? -1 : 1;

  if (x < 0)
    x = -x;
  while (x >>= 1)
    ++logvalue;
  return sign * logvalue;
}

void doSortedLogOffsetHistogram(Graph& graph) {
  // Graph copy;
  // {
  //   // Original FileGraph is immutable because it is backed by a file
  //   copy = graph;
  // }

  // std::vector<std::map<int, size_t> > hists;
  // hists.emplace_back();
  // auto hist = &hists.back();
  // int curHist = 0;
  // auto p = galois::block_range(
  //     boost::counting_iterator<size_t>(0),
  //     boost::counting_iterator<size_t>(graph.sizeEdges()),
  //     curHist,
  //     numHist);
  // for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
  //   copy.sortEdges<void>(*ii, EdgeComp());

  //   GNode last = 0;
  //   bool first = true;
  //   for (auto jj = copy.edge_begin(*ii), ej = copy.edge_end(*ii); jj != ej; ++jj) {
  //     GNode dst = copy.getEdgeDst(jj);
  //     ptrdiff_t diff = dst - (ptrdiff_t) last;

  //     if (!first) {
  //       int index = getLogIndex(diff);
  //       ++(*hist)[index];
  //     }
  //     first = false;
  //     last = dst;
  //     if (++p.first == p.second) {
  //       hists.emplace_back();
  //       hist = &hists.back();
  //       curHist += 1;
  //       p = galois::block_range(
  //           boost::counting_iterator<size_t>(0),
  //           boost::counting_iterator<size_t>(graph.sizeEdges()),
  //           curHist,
  //           numHist);
  //     }
  //   }
  // }

  // printHistogram("LogOffset", hists);
}

void doDestinationHistogram(Graph& graph) {
  std::map<uint64_t, uint64_t> hist;
  for (auto ii : graph)
    for (auto jj : graph.edges(ii))
      ++hist[graph.getEdgeDst(jj)];
  printHistogram("DestinationBin", hist);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  try {
    Graph graph(inputfilename);
    for (unsigned i = 0; i != statModeList.size(); ++i) {
      switch (statModeList[i]) {
      case degreehist: doDegreeHistogram(graph); break;
      case degrees: doDegrees(graph); break;
      case maxDegreeNode: findMaxDegreeNode(graph); break;
      case dsthist: doDestinationHistogram(graph); break;
      case indegreehist: doInDegreeHistogram(graph); break;
      case sortedlogoffsethist: doSortedLogOffsetHistogram(graph); break;
      case sparsityPattern: {
        unsigned lastrow = ~0;
        doSparsityPattern(graph, [&lastrow] (unsigned x, unsigned y, bool val) {if (y != lastrow) { lastrow = y; std::cout << '\n'; } std::cout << (val ? 'x' : '.'); } );
        std::cout << '\n';
        break;
      }
      case summary: doSummary(graph); break;
      default:  std::cerr << "Unknown stat requested\n"; break;
      }
    }
    return 0;
  } catch (...) {
    std::cerr << "failed\n";
    return 1;
  }
}
