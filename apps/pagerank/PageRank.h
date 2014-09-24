/** Page rank application -*- C++ -*-
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
 * @author Joyce Whang <joyce@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */


#ifndef APPS_PAGERANK_PAGERANK_H
#define APPS_PAGERANK_PAGERANK_H

#include "llvm/Support/CommandLine.h"

//! d is the damping factor. Alpha is the prob that user will do a random jump, i.e., 1 - d
//static const float alpha = 1.0 - 0.85;
static const float alpha = 0.85; // Joyce changed to this which is a usual way to define alpha.

typedef double PRTy;

template<typename Graph>
unsigned nout(Graph& g, typename Graph::GraphNode n, Galois::MethodFlag flag) {
  return std::distance(g.edge_begin(n, flag), g.edge_end(n, flag));
}

template<typename Graph>
double computePageRankInOut(Graph& g, typename Graph::GraphNode src, int prArg, Galois::MethodFlag lockflag) {
  double sum = 0;
  for (auto jj = g.in_edge_begin(src, lockflag), ej = g.in_edge_end(src, lockflag);
       jj != ej; ++jj) {
    auto dst = g.getInEdgeDst(jj);
    auto& ddata = g.getData(dst, lockflag);
    sum += ddata.getPageRank(prArg) / nout(g, dst, lockflag);
  }
  return sum;
}

PRTy atomicAdd(std::atomic<PRTy>& v, PRTy delta) {
  PRTy old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}

template<typename Graph>
void verifyInOut(Graph& graph, PRTy tolerance) {
  for(auto N : graph) {
    auto& data = graph.getData(N);
    auto residual = data.value - (alpha*computePageRankInOut(graph, N, 0, Galois::MethodFlag::NONE) + (1.0 - alpha));
    if (residual > tolerance) {
      std::cout << N << " residual " << residual << " pr " << data.getPageRank(0) << " data " << data << "\n";
    }
  }
}


template<typename Graph, typename InnerGraph>
void check_types() {
  static_assert(std::is_same<typename std::iterator_traits<typename Graph::edge_iterator>::iterator_category,
                std::random_access_iterator_tag>::value, "Not random");
  static_assert(std::is_same<typename std::iterator_traits<typename InnerGraph::edge_iterator>::iterator_category,
                std::random_access_iterator_tag>::value, "Not random");
  static_assert(std::is_same<typename std::iterator_traits<typename Graph::in_edge_iterator>::iterator_category,
                std::random_access_iterator_tag>::value, "Not random");
}


#endif
