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
 * @author Yi-Shan Lu <yishanlu@utexas.edu>
 */


#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/TypeTraits.h"
#include "PageRank.h"

#include <atomic>
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>


static const float alpha = 0.85; 
typedef double PRTy;

struct Initialization {
  Graph& g;

  void operator()(const GNode n) {
    auto& data = g.getData(n);
    data.DAd.vDouble = 1.0 - alpha; // value

    data.DAd.vAtomicDouble = 0.0;   // residual
  }
};

//! Make values unique
template<typename GNode>
struct TopPair {
  double value;
  GNode id;

  TopPair(double v, GNode i): value(v), id(i) { }

  bool operator<(const TopPair& b) const {
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};



template<typename Graph>
static NodeDouble *reportTop(Graph& graph, int topn, const ValAltTy result) {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair,GNode> Top;

  // normalize the PageRank value so that the sum is equal to one
  double sum=0.0;
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    node_data_reference n = graph.getData(src);
    sum += n.DAd.vDouble;
  }

  Top top;
  
  //std::cout<<"print PageRank\n";
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    node_data_reference n = graph.getData(src);
    double value = n.DAd.vDouble/sum; // normalized PR (divide PR by sum)
    //float value = n.getPageRank(); // raw PR 
    //std::cout<<value<<" "; 
    Pair key(value, src);

    if ((int) top.size() < topn) {
      top.insert(std::make_pair(key, src));
      continue;
    }

    if (top.begin()->first < key) {
      top.erase(top.begin());
      top.insert(std::make_pair(key, src));
    }

    n.attr[result] = std::to_string(value);
  }
  //std::cout<<"\nend of print\n";

  NodeDouble *pr = new NodeDouble [topn] ();
  int rank = 0;
  for (typename Top::reverse_iterator ii = top.rbegin(), ei = top.rend(); ii != ei; ++ii, ++rank) {
    pr[rank].n = ii->first.id;
    pr[rank].v = ii->first.value;
  }
  return pr;

#if 0
  int rank = 1;
  std::cout << "Rank PageRank Id\n";
  for (typename Top::reverse_iterator ii = top.rbegin(), ei = top.rend(); ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
#endif
}

PRTy atomicAdd(std::atomic<PRTy>& v, PRTy delta) {
  PRTy old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}


struct PageRank {
  Graph& graph;
  PRTy tolerance;
  
  PageRank(Graph& g, PRTy t) : graph(g), tolerance(t) {}
  
  void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
    auto& sdata = graph.getData(src);      
    auto& sResidual = sdata.DAd.vAtomicDouble;
    auto& sValue = sdata.DAd.vDouble;
    galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    
    if (std::abs(sResidual) > tolerance) {
      PRTy oldResidual = sResidual.exchange(0.0);
      sValue += oldResidual;
      int src_nout = std::distance(graph.edge_begin(src, flag), graph.edge_end(src,flag));
      PRTy delta = oldResidual*alpha/src_nout;
      // for each out-going neighbors
      for (auto jj : graph.edges(src, flag)) {
        GNode dst = graph.getEdgeDst(jj);
        auto& ddata = graph.getData(dst, flag);
        auto& dResidual = ddata.DAd.vAtomicDouble;
        auto old = atomicAdd(dResidual, delta);
        if (std::abs(old) <= tolerance && std::abs(old + delta) >= tolerance)
          ctx.push(dst);
      }
    }
  }
};


void initResidual(Graph& graph) {

  //use residual for the partial, scaled initial residual
  galois::do_all_local(graph, [&graph] (const typename Graph::GraphNode& src) {
      //contribute residual
      auto nout = std::distance(graph.edge_begin(src), graph.edge_end(src));
      for (auto ii : graph.edges(src)) {
        auto dst = graph.getEdgeDst(ii);
        auto& ddata = graph.getData(dst);
        auto& dResidual = ddata.DAd.vAtomicDouble;
        atomicAdd(dResidual, 1.0/nout);
      }
    }, galois::steal<true>());
  //scale residual
  galois::do_all_local(graph, [&graph] (const typename Graph::GraphNode& src) {
      auto& data = graph.getData(src);
      auto& dResidual = data.DAd.vAtomicDouble;
      dResidual = dResidual * alpha * (1.0-alpha);
    }, galois::steal<true>());
}

NodeDouble *analyzePagerank(Graph *g, int topK, double tolerance, const ValAltTy result) {
//  galois::StatManager statManager;

//  galois::StatTimer T("OverheadTime");
//  T.start();

//  std::cout << "Running Edge Async version\n";
//  std::cout << "tolerance: " << tolerance << "\n";
  galois::do_all_local(*g, Initialization{*g});
  initResidual(*g);

//  galois::StatTimer Tmain;
//  Tmain.start();  
  typedef galois::worklists::dChunkedFIFO<256> WL;
  galois::for_each_local(*g, PageRank{*g, tolerance}, galois::wl<WL>());
//  Tmain.stop();
  
//  T.stop();

  return reportTop(*g, topK, result);
}

