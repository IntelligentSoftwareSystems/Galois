/** ?? -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 */

/**
 * Common support for worklist experiments.
 */
#ifndef GALOIS_PRIORITYSCHEDULING_WORKLIST_H
#define GALOIS_PRIORITYSCHEDULING_WORKLIST_H

#include "Galois/WorkList/WorkList.h"
#include "Galois/WorkList/WorkListExperimental.h"
#include "Galois/gIO.h"

namespace Exp {

__attribute__((weak)) llvm::cl::opt<std::string> WorklistName("wl", llvm::cl::desc("Worklist to use"), llvm::cl::init("DEFAULT"));

//FIXME: using in a header
using namespace galois::Runtime;
using namespace galois::Substrate;
using namespace galois::worklists;

template<int CS, bool LF>
struct PickInner;

template<int CS>
struct PickInner<CS, true> {
  typedef dChunkedLIFO<CS> dChunk;
  typedef  ChunkedLIFO<CS> Chunk;
};
template<int CS>
struct PickInner<CS, false> {
  typedef dChunkedFIFO<CS> dChunk;
  typedef  ChunkedFIFO<CS> Chunk;
};

 template<int ChunkSize, typename Ind, typename DEFAULT, typename Less, typename Greater, bool LF = false >
struct PriAuto {

   typedef typename PickInner<ChunkSize, LF>::dChunk dChunk;
   typedef typename PickInner<ChunkSize, LF>::Chunk   Chunk;

  //OBIM
  typedef   OrderedByIntegerMetric<Ind,dChunk, true> OBIM_DMB;
  typedef CTOrderedByIntegerMetric<Ind,dChunk, true> OBIM_DSB;
  typedef   OrderedByIntegerMetric<Ind, Chunk, true> OBIM_CMB;
  typedef CTOrderedByIntegerMetric<Ind, Chunk, true> OBIM_CSB;
  typedef   OrderedByIntegerMetric<Ind,dChunk,false> OBIM_DMN;
  typedef CTOrderedByIntegerMetric<Ind,dChunk,false> OBIM_DSN;
  typedef   OrderedByIntegerMetric<Ind, Chunk,false> OBIM_CMN;
  typedef CTOrderedByIntegerMetric<Ind, Chunk,false> OBIM_CSN;

  //TBB
#ifdef USE_TBB
  typedef TbbPriQueue<Greater> TBB;
  typedef PTbb<Greater> PTBB;
  typedef STbb<Greater> STBB;
#endif
  //MISC
  typedef SkipListQueue<Less> SLQ;
  typedef SetQueue<Less> SETQ;

  template<typename IterTy, typename FunctionTy, typename... Args>
  static void for_each(IterTy b, IterTy e, FunctionTy f, Args... args) {
    static bool printed = false;
#define WLFOO2(__x)							\
    if (WorklistName == #__x) {						\
      if (!printed) {							\
        galois::gInfo("WorkList ", #__x);                    \
	printed = true;							\
      }									\
      galois::for_each(b,e,f,std::forward<Args>(args)..., galois::wl<__x>()); \
    } else
#include "PrioritySchedulers.h"
#undef WLFOO2
#define WLFOO2(__x)							\
    if (WorklistName == "NI_" #__x) {					\
      if (!printed) {							\
        galois::gInfo("WorkList ", "NI_" #__x);              \
	printed = true;							\
      }									\
      galois::for_each(b,e,f,std::forward<Args>(args)..., galois::wl<NoInlineFilter<__x>>()); \
    } else
#include "PrioritySchedulers.h"
#undef WLFOO2

    {
      GALOIS_DIE("unknown worklist: ", WorklistName.c_str(), "\n");
    }
  }
  template<typename InitItemTy, typename FunctionTy, typename... Args>
  static void for_each(InitItemTy i, FunctionTy f, Args... args) {
    InitItemTy wl[1] = {i};
    for_each(&wl[0], &wl[1], f, std::forward<Args>(args)...);
  }
};

} // end namespace

#endif
