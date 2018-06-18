/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

/**
 * Common support for worklist experiments.
 */
#ifndef GALOIS_PRIORITYSCHEDULING_WORKLIST_H
#define GALOIS_PRIORITYSCHEDULING_WORKLIST_H

#include "galois/worklists/WorkList.h"
#include "galois/worklists/WorkListExperimental.h"
#include "galois/gIO.h"

namespace Exp {

__attribute__((weak)) llvm::cl::opt<std::string>
    WorklistName("wl", llvm::cl::desc("Worklist to use"),
                 llvm::cl::init("DEFAULT"));

// FIXME: using in a header
using namespace galois::runtime;
using namespace galois::substrate;
using namespace galois::worklists;

template <int CS, bool LF>
struct PickInner;

template <int CS>
struct PickInner<CS, true> {
  typedef PerSocketChunkLIFO<CS> PSchunk;
  typedef ChunkLIFO<CS> Chunk;
};
template <int CS>
struct PickInner<CS, false> {
  typedef PerSocketChunkFIFO<CS> PSchunk;
  typedef ChunkFIFO<CS> Chunk;
};

template <int ChunkSize, typename Ind, typename DEFAULT, typename Less,
          typename Greater, bool LF = false>
struct PriAuto {

  typedef typename PickInner<ChunkSize, LF>::PSchunk PSchunk;
  typedef typename PickInner<ChunkSize, LF>::Chunk Chunk;

  // OBIM
  typedef OrderedByIntegerMetric<Ind, PSchunk, true> OBIM_DMB;
  typedef CTOrderedByIntegerMetric<Ind, PSchunk, true> OBIM_DSB;
  typedef OrderedByIntegerMetric<Ind, Chunk, true> OBIM_CMB;
  typedef CTOrderedByIntegerMetric<Ind, Chunk, true> OBIM_CSB;
  typedef OrderedByIntegerMetric<Ind, PSchunk, false> OBIM_DMN;
  typedef CTOrderedByIntegerMetric<Ind, PSchunk, false> OBIM_DSN;
  typedef OrderedByIntegerMetric<Ind, Chunk, false> OBIM_CMN;
  typedef CTOrderedByIntegerMetric<Ind, Chunk, false> OBIM_CSN;

  // TBB
#ifdef USE_TBB
  typedef TbbPriQueue<Greater> TBB;
  typedef PTbb<Greater> PTBB;
  typedef STbb<Greater> STBB;
#endif
  // MISC
  typedef SkipListQueue<Less> SLQ;
  typedef SetQueue<Less> SETQ;

  template <typename IterTy, typename FunctionTy, typename... Args>
  static void for_each(IterTy b, IterTy e, FunctionTy f, Args... args) {
    static bool printed = false;
#define WLFOO2(__x)                                                            \
  if (WorklistName == #__x) {                                                  \
    if (!printed) {                                                            \
      galois::gInfo("WorkList ", #__x);                                        \
      printed = true;                                                          \
    }                                                                          \
    galois::for_each(galois::iterate(b, e), f, std::forward<Args>(args)...,    \
                     galois::wl<__x>());                                       \
  } else
#include "PrioritySchedulers.h"
#undef WLFOO2
#define WLFOO2(__x)                                                            \
  if (WorklistName == "NI_" #__x) {                                            \
    if (!printed) {                                                            \
      galois::gInfo("WorkList ", "NI_" #__x);                                  \
      printed = true;                                                          \
    }                                                                          \
    galois::for_each(b, e, f, std::forward<Args>(args)...,                     \
                     galois::wl<NoInlineFilter<__x>>());                       \
  } else
#include "PrioritySchedulers.h"
#undef WLFOO2

    { GALOIS_DIE("unknown worklist: ", WorklistName.c_str(), "\n"); }
  }
  template <typename InitItemTy, typename FunctionTy, typename... Args>
  static void for_each(InitItemTy i, FunctionTy f, Args... args) {
    InitItemTy wl[1] = {i};
    for_each(&wl[0], &wl[1], f, std::forward<Args>(args)...);
  }
};

} // namespace Exp

#endif
