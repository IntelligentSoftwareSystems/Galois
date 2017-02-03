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

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <vector>
#include <cstdlib>

namespace cll = llvm::cl;

enum StatMode {
  degreehist,
  degrees,
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
      clEnumVal(dsthist, "Histogram of destinations"),
      clEnumVal(indegreehist, "Histogram of indegrees"),
      clEnumVal(sortedlogoffsethist, "Histogram of neighbor offsets with sorted edges"),
      clEnumVal(sparsityPattern, "Pattern of non-zeros when graph is interpreted as a sparse matrix"),
      clEnumVal(summary, "Graph summary"),
      clEnumValEnd));
static cll::opt<int> numHist("numHist", cll::desc("Number of histograms to bin input over"), cll::init(1));
static cll::opt<int> numDstBins("numDstBins", cll::desc("Number of bins to place destinations in"), cll::init(100));

typedef Galois::Graph::FileGraph Graph;
typedef Graph::GraphNode GNode;
Graph graph;

void doSummary() {
  std::cout << "NumNodes: " << graph.size() << "\n";
  std::cout << "NumEdges: " << graph.sizeEdges() << "\n";
  std::cout << "SizeofEdge: " << graph.edgeSize() << "\n";
}

void doDegrees() {
  for (auto n : graph) {
    std::cout << std::distance(graph.neighbor_begin(n), graph.neighbor_end(n)) << "\n";
  }
}

template<typename HistsTy>
void printHistogram(const std::string& name, HistsTy& hists) {
  typedef typename HistsTy::value_type::key_type KeyType;

  int width = 0;
  for (auto ii = hists.begin(), ei = hists.end(); ii != ei; ++ii) {
    if (ii->rbegin() != ii->rend()) {
      KeyType most = ii->rbegin()->first;
      if (most)
        width = std::max(width, (int) std::ceil(std::log10(most)));
    }
  }
  int bin = 0;
  int bwidth = (int) std::ceil(std::log10((int) numHist));
  std::cout << "Hist," << name << ",Count\n";
  for (auto ii = hists.begin(), ei = hists.end(); ii != ei; ++ii, ++bin) {
    for (auto pp = ii->begin(), ep = ii->end(); pp != ep; ++pp) {
      std::cout.width(bwidth);
      std::cout << bin << ",";
      if (width)
        std::cout.width(width);
      std::cout << pp->first << "," << pp->second << "\n";
    }
  }
}

void doSparsityPattern() {
  int columns = 80;
  // COLUMNS is a special variable in bash (it executes tput cols) so users
  // need to explicitly set the COLUMNS environment variable for this info
  // to be passed here
  const char* colStr = std::getenv("COLUMNS");
  if (colStr)
    columns = std::atoi(colStr);
  unsigned blockSize = (graph.size() + columns - 1) / columns;

  for (int i = 0; i < columns; ++i) {
    std::vector<bool> row(columns);
    auto p = Galois::Runtime::block_range(graph.begin(), graph.end(), i, columns);
    for (auto ii = p.first, ei = p.second; ii != ei; ++ii) {
      for (auto jj : graph.out_edges(*ii)) {
        row[graph.getEdgeDst(jj) / blockSize] = true;
      }
    }
    for (auto rr : row) {
      if (rr)
        std::cout << "x";
      else
        std::cout << ".";
    }
    std::cout << "\n";
  }
}

void doDegreeHistogram() {
  unsigned numEdges = 0;

  std::vector<std::map<unsigned, unsigned> > hists;
  for (int i = 0; i < numHist; ++i) {
    auto p = Galois::Runtime::block_range(graph.begin(), graph.end(), i, numHist);

    hists.emplace_back();
    auto& hist = hists.back();
    for (auto ii = p.first, ei = p.second; ii != ei; ++ii) {
      unsigned val = std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii));
      numEdges += val;
      ++hist[val];
    }
  }

  printHistogram("Degree", hists);
}

void doInDegreeHistogram() {
  std::vector<std::map<GNode, unsigned> > invs;
  std::vector<std::map<unsigned, unsigned> > hists;

  for (int i = 0; i < numHist; ++i) {
    auto p = Galois::Runtime::block_range(graph.begin(), graph.end(), i, numHist);

    hists.emplace_back();
    invs.emplace_back();
    auto& hist = hists.back();
    auto& inv = invs.back();
    for (auto ii = p.first, ei = p.second; ii != ei; ++ii) {
      for (auto jj = graph.edge_begin(*ii), ej = graph.edge_end(*ii); jj != ej; ++jj)
        ++inv[graph.getEdgeDst(jj)];
    for (auto pp = inv.begin(), ep = inv.end(); pp != ep; ++pp)
      ++hist[pp->second];
    }
  }

  printHistogram("InDegree", hists);
}

struct EdgeComp {
  typedef Galois::Graph::EdgeSortValue<GNode, void> Edge;
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

void doSortedLogOffsetHistogram() {
  Graph copy;
  {
    // Original FileGraph is immutable because it is backed by a file
    copy = graph;
  }

  std::vector<std::map<int, size_t> > hists;
  hists.emplace_back();
  auto hist = &hists.back();
  int curHist = 0;
  auto p = Galois::Runtime::block_range(
      boost::counting_iterator<size_t>(0),
      boost::counting_iterator<size_t>(graph.sizeEdges()),
      curHist,
      numHist);
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    copy.sortEdges<void>(*ii, EdgeComp());

    GNode last = 0;
    bool first = true;
    for (auto jj = copy.edge_begin(*ii), ej = copy.edge_end(*ii); jj != ej; ++jj) {
      GNode dst = copy.getEdgeDst(jj);
      ptrdiff_t diff = dst - (ptrdiff_t) last;

      if (!first) {
        int index = getLogIndex(diff);
        ++(*hist)[index];
      }
      first = false;
      last = dst;
      if (++p.first == p.second) {
        hists.emplace_back();
        hist = &hists.back();
        curHist += 1;
        p = Galois::Runtime::block_range(
            boost::counting_iterator<size_t>(0),
            boost::counting_iterator<size_t>(graph.sizeEdges()),
            curHist,
            numHist);
      }
    }
  }

  printHistogram("LogOffset", hists);
}

void doDestinationHistogram() {
  std::vector<std::map<unsigned, size_t> > hists;
  hists.emplace_back();
  auto hist = &hists.back();
  int curHist = 0;
  auto p = Galois::Runtime::block_range(
      boost::counting_iterator<size_t>(0),
      boost::counting_iterator<size_t>(graph.sizeEdges()),
      curHist,
      numHist);
  size_t blockSize = (graph.size() + numDstBins - 1) / numDstBins;
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    for (auto jj = graph.edge_begin(*ii), ej = graph.edge_end(*ii); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      ++(*hist)[dst / blockSize];

      if (++p.first == p.second) {
        hists.emplace_back();
        hist = &hists.back();
        curHist += 1;
        p = Galois::Runtime::block_range(
            boost::counting_iterator<size_t>(0),
            boost::counting_iterator<size_t>(graph.sizeEdges()),
            curHist,
            numHist);
      }
    }
  }

  printHistogram("DestinationBin", hists);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  graph.fromFile(inputfilename);
  
  for (unsigned i = 0; i != statModeList.size(); ++i) {
    switch (statModeList[i]) {
      case degreehist: doDegreeHistogram(); break;
      case degrees: doDegrees(); break;
      case dsthist: doDestinationHistogram(); break;
      case indegreehist: doInDegreeHistogram(); break;
      case sortedlogoffsethist: doSortedLogOffsetHistogram(); break;
      case sparsityPattern: doSparsityPattern(); break;
      case summary: doSummary(); break;
      default: abort(); break;
    }
  }

  return 0;
}
