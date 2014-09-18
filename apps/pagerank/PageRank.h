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
static const float alpha = 1.0 - 0.85;
static const float alpha2 = 0.85; // Joyce changed to this which is a usual way to define alpha.

//! maximum relative change until we deem convergence
static const float tolerance = 0.01; 
//static const float tolerance = 0.0001; // Joyce

//ICC v13.1 doesn't yet support std::atomic<float> completely, emmulate its
//behavor with std::atomic<int>
struct atomic_float : public std::atomic<int> {
  static_assert(sizeof(int) == sizeof(float), "int and float must be the same size");

  float atomicIncrement(float value) {
    while (true) {
      union { float as_float; int as_int; } oldValue = { read() };
      union { float as_float; int as_int; } newValue = { oldValue.as_float + value };
      if (this->compare_exchange_strong(oldValue.as_int, newValue.as_int))
        return newValue.as_float;
    }
  }

  float read() {
    union { int as_int; float as_float; } caster = { this->load(std::memory_order_relaxed) };
    return caster.as_float;
  }

  void write(float v) {
    union { float as_float; int as_int; } caster = { v };
    this->store(caster.as_int, std::memory_order_relaxed);
  }
};

extern llvm::cl::opt<unsigned int> memoryLimit;
extern llvm::cl::opt<std::string> filename;
extern llvm::cl::opt<unsigned int> maxIterations;

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
