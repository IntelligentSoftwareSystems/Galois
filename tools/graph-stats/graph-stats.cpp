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
#include "Galois/Graph/LCGraph.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <vector>

namespace cll = llvm::cl;

enum StatMode {
  degreehist,
  degrees,
  dsthist,
  indegreehist,
  sortedlogoffsethist,
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
      clEnumVal(summary, "Graph summary"),
      clEnumValEnd));
static cll::opt<int> numHist("numHist", cll::desc("Number of histograms to bin input over"), cll::init(1));
static cll::opt<int> numDstBins("numDstBins", cll::desc("Number of bins to place destinations in"), cll::init(100));

typedef Galois::Graph::FileGraph Graph;
typedef Graph::GraphNode GNode;
Graph graph;

void do_summary() {
  std::cout << "NumNodes: " << graph.size() << "\n";
  std::cout << "NumEdges: " << graph.sizeEdges() << "\n";
  std::cout << "SizeofEdge: " << graph.edgeSize() << "\n";
}

void do_degrees() {
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    std::cout << std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii)) << "\n";
  }
}

template<typename HistsTy>
void print_hists(const std::string& name, HistsTy& hists) {
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

void do_degreehist() {
  unsigned numEdges = 0;

  std::vector<std::map<unsigned, unsigned> > hists;
  for (int i = 0; i < numHist; ++i) {
    auto p = Galois::block_range(graph.begin(), graph.end(), i, numHist);

    hists.emplace_back();
    auto& hist = hists.back();
    for (auto ii = p.first, ee = p.second; ii != ee; ++ii) {
      unsigned val = std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii));
      numEdges += val;
      ++hist[val];
    }
  }

  print_hists("Degree", hists);
}

void do_indegreehist() {
  std::vector<std::map<GNode, unsigned> > invs;
  std::vector<std::map<unsigned, unsigned> > hists;

  for (int i = 0; i < numHist; ++i) {
    auto p = Galois::block_range(graph.begin(), graph.end(), i, numHist);

    hists.emplace_back();
    invs.emplace_back();
    auto& hist = hists.back();
    auto& inv = invs.back();
    for (auto ii = p.first, ee = p.second; ii != ee; ++ii) {
      for (auto jj = graph.edge_begin(*ii), ej = graph.edge_end(*ii); jj != ej; ++jj)
        ++inv[graph.getEdgeDst(jj)];
    for (auto pp = inv.begin(), ep = inv.end(); pp != ep; ++pp)
      ++hist[pp->second];
    }
  }

  print_hists("InDegree", hists);
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

void do_sortedlogoffsethist() {
  Graph copy;
  {
    // Original FileGraph is immutable because it is backed by a file
    copy = graph;
  }

  std::vector<std::map<int, size_t> > hists;
  hists.emplace_back();
  auto hist = &hists.back();
  int curHist = 0;
  auto p = Galois::block_range(
      boost::counting_iterator<size_t>(0),
      boost::counting_iterator<size_t>(graph.sizeEdges()),
      curHist,
      numHist);
  for (auto ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
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
        p = Galois::block_range(
            boost::counting_iterator<size_t>(0),
            boost::counting_iterator<size_t>(graph.sizeEdges()),
            curHist,
            numHist);
      }
    }
  }

  print_hists("LogOffset", hists);
}

void do_dsthist() {
  std::vector<std::map<unsigned, size_t> > hists;
  hists.emplace_back();
  auto hist = &hists.back();
  int curHist = 0;
  auto p = Galois::block_range(
      boost::counting_iterator<size_t>(0),
      boost::counting_iterator<size_t>(graph.sizeEdges()),
      curHist,
      numHist);
  size_t blockSize = (graph.size() + numDstBins - 1) / numDstBins;
  for (auto ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
    for (auto jj = graph.edge_begin(*ii), ej = graph.edge_end(*ii); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      ++(*hist)[dst / blockSize];

      if (++p.first == p.second) {
        hists.emplace_back();
        hist = &hists.back();
        curHist += 1;
        p = Galois::block_range(
            boost::counting_iterator<size_t>(0),
            boost::counting_iterator<size_t>(graph.sizeEdges()),
            curHist,
            numHist);
      }
    }
  }

  print_hists("DestinationBin", hists);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  graph.structureFromFile(inputfilename);
  
  for (unsigned i = 0; i != statModeList.size(); ++i) {
    switch (statModeList[i]) {
    case degreehist: do_degreehist(); break;
    case degrees: do_degrees(); break;
    case dsthist: do_dsthist(); break;
    case indegreehist: do_indegreehist(); break;
    case sortedlogoffsethist: do_sortedlogoffsethist(); break;
    case summary: do_summary(); break;
    default: abort(); break;
    }
  }

  return 0;
}
