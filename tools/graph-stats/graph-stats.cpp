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
  summary,
  degrees,
  degreehist,
  indegreehist
};

static cll::opt<std::string> inputfilename(cll::Positional, cll::desc("<graph file>"), cll::Required);
static cll::list<StatMode> statModeList(cll::desc("Available stats:"),
    cll::values(
      clEnumVal(summary, "Graph summary"),
      clEnumVal(degrees, "Node degrees"),
      clEnumVal(degreehist, "Histogram of degrees"),
      clEnumVal(indegreehist, "Histogram of indegrees"),
      clEnumValEnd));

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

void do_hist() {
  unsigned numEdges = 0;
  std::map<unsigned, unsigned> hist;
  for (auto ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
    unsigned val = std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii));
    numEdges += val;
    ++hist[val];
  }
  for (auto pp = hist.begin(), ep = hist.end(); pp != ep; ++pp)
    std::cout << pp->first << " : " << pp->second << "\n";
}

void do_inhist() {
  //unsigned numEdges = 0;
  std::map<GNode, unsigned> inv;
  std::map<unsigned, unsigned> hist;
  for (auto ii = graph.begin(), ee = graph.end(); ii != ee; ++ii)
    for (auto ei = graph.edge_begin(*ii), eie = graph.edge_end(*ii); ei != eie; ++ei)
      ++inv[graph.getEdgeDst(ei)];
  for (auto pp = inv.begin(), ep = inv.end(); pp != ep; ++pp)
    ++hist[pp->second];
  for (auto pp = hist.begin(), ep = hist.end(); pp != ep; ++pp)
    std::cout << pp->first << " : " << pp->second << "\n";
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  graph.structureFromFile(inputfilename);
  
  for (unsigned i = 0; i != statModeList.size(); ++i) {
    switch (statModeList[i]) {
    case summary: do_summary(); break;
    case degrees: do_degrees(); break;
    case degreehist: do_hist(); break;
    case indegreehist: do_inhist(); break;
    default: abort(); break;
    }
  }

  return 0;
}
