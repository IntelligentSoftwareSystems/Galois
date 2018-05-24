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

static const float alpha = 0.85; 
extern bool outOnly;

typedef double PRTy;

template<typename Graph>
unsigned nout(Graph& g, typename Graph::GraphNode n, galois::MethodFlag flag) {
  return std::distance(g.edge_begin(n, flag), g.edge_end(n, flag));
}

template<typename Graph>
unsigned ninout(Graph& g, typename Graph::GraphNode n, galois::MethodFlag flag) {
  return std::distance(g.in_edge_begin(n, flag), g.in_edge_end(n, flag)) + nout(g, n, flag);
}

template<typename Graph>
double computePageRankInOut(Graph& g, typename Graph::GraphNode src, int prArg, galois::MethodFlag lockflag) {
  double sum = 0;
  for (auto jj = g.in_edge_begin(src, lockflag), ej = g.in_edge_end(src, lockflag); jj != ej; ++jj) {
    auto dst = g.getInEdgeDst(jj);
    auto& ddata = g.getData(dst, lockflag);
    sum += ddata.getPageRank(prArg) / nout(g, dst, lockflag);
  }
  return alpha*sum + (1.0 - alpha);
}

template<typename Graph>
void initResidual(Graph& graph) {
  galois::do_all(graph, [&graph] (const typename Graph::GraphNode& src) {
      auto& data = graph.getData(src);
      // for each in-coming neighbour, add residual
      PRTy sum = 0.0;
      for (auto jj = graph.in_edge_begin(src), ej = graph.in_edge_end(src); jj != ej; ++jj){
        auto dst = graph.getInEdgeDst(jj);
        auto& ddata = graph.getData(dst);
        sum += 1.0/nout(graph,dst, galois::MethodFlag::UNPROTECTED);  
      }
      data.residual = sum * alpha * (1.0-alpha);
    }, galois::steal());
}

template<typename Graph, typename PriFn>
void initResidual(Graph& graph, galois::InsertBag<std::pair<typename Graph::GraphNode, int> >& b, const PriFn& pri) {
  galois::do_all(graph, [&graph, &b, &pri] (const typename Graph::GraphNode& src) {
      auto& data = graph.getData(src);
      // for each in-coming neighbour, add residual
      PRTy sum = 0.0;
      for (auto jj = graph.in_edge_begin(src), ej = graph.in_edge_end(src); jj != ej; ++jj){
        auto dst = graph.getInEdgeDst(jj);
        auto& ddata = graph.getData(dst);
        sum += 1.0/nout(graph,dst, galois::MethodFlag::UNPROTECTED);  
      }
      data.residual = sum * alpha * (1.0-alpha);
      b.push(std::make_pair(src, pri(graph, src)));
    }, galois::steal());
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
    auto residual = std::fabs(data.getPageRank() - computePageRankInOut(graph, N, 1, galois::MethodFlag::UNPROTECTED));
    if (residual > tolerance) {
      std::cout << N << " residual " << residual << " pr " << data.getPageRank() << " data " << data << "\n";
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
