/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "Metis.h"
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/gstl.h"

#include <iostream>

namespace {

using MatchingPolicy    = GNode(GNode, GGraph*);
using MatchingSubPolicy = std::pair<GNode, int>(GNode, GGraph*, bool tag);

std::pair<GNode, int> HEMmatch(GNode node, GGraph* graph, bool) {
  GNode retval = node; // match self if nothing else
  int maxwgt   = std::numeric_limits<int>::min();
  //    nume += std::distance(graph->edge_begin(node), graph->edge_end(node));
  for (auto jj : graph->edges(node, galois::MethodFlag::UNPROTECTED)) {
    //      ++checked;
    GNode neighbor = graph->getEdgeDst(jj);
    MetisNode& neighMNode =
        graph->getData(neighbor, galois::MethodFlag::UNPROTECTED);
    int edgeData = graph->getEdgeData(jj, galois::MethodFlag::UNPROTECTED);
    if (!neighMNode.isMatched() && neighbor != node && maxwgt < edgeData) {
      maxwgt = edgeData;
      retval = neighbor;
    }
  }
  return std::make_pair(retval, maxwgt);
}
GNode HEMmatch(GNode node, GGraph* graph) {
  return HEMmatch(node, graph, true).first;
}

GNode RMmatch(GNode node, GGraph* graph) {
  for (auto jj : graph->edges(node, galois::MethodFlag::UNPROTECTED)) {
    GNode neighbor = graph->getEdgeDst(jj);
    if (!graph->getData(neighbor, galois::MethodFlag::UNPROTECTED)
             .isMatched() &&
        neighbor != node)
      return neighbor;
  }
  return node;
  // Don't actually do random, just choose first
}
// std::pair<GNode, int> RMmatch(GNode node, GGraph* graph, bool tag) {
//  return std::make_pair(RMmatch(node, graph), 0);
//}

template <MatchingSubPolicy matcher>
GNode TwoHopMatcher(GNode node, GGraph* graph) {
  std::pair<GNode, int> retval(node, std::numeric_limits<int>::min());
  for (auto jj : graph->edges(node, galois::MethodFlag::UNPROTECTED)) {
    GNode neighbor             = graph->getEdgeDst(jj);
    std::pair<GNode, int> tval = matcher(neighbor, graph, true);
    if (tval.first != node && tval.first != neighbor &&
        tval.second > retval.second)
      retval = tval;
  }
  return retval.first;
}

typedef galois::GAccumulator<unsigned> Pcounter;

/*
 *This function is responsible for matching.
 1. There are two types of matching. Random and Heavy Edge matching
 2. Random matching picks any random node above a threshold and matches the
 nodes. RM.h
 3. Heavy Edge Matching matches the vertex which is connected by the heaviest
 edge. HEM.h
 4. This function can also create the multinode, i.e. the node which is created
 on combining two matched nodes.
 5. You can enable/disable 4th by changing variantMetis::mergeMatching
*/
template <MatchingPolicy matcher, typename WL>
void parallelMatchAndCreateNodes(MetisGraph* graph, Pcounter& pc,
                                 GNodeBag& noEdgeBag, bool selfMatch) {
  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);

  galois::for_each(
      galois::iterate(*fineGGraph),
      [&](GNode item, galois::UserContext<GNode>&) {
        if (fineGGraph->getData(item).isMatched())
          return;

        if (fineGGraph->edge_begin(item, galois::MethodFlag::UNPROTECTED) ==
            fineGGraph->edge_end(item, galois::MethodFlag::UNPROTECTED)) {
          noEdgeBag.push(item);
          return;
        }

        GNode ret;
        do {
          ret = matcher(item, fineGGraph);
          // lock ret, since we found it lock-free it may be matched, so try
          // again
        } while (fineGGraph->getData(ret).isMatched());

        // at this point both ret and item (and failed matches) are locked.
        // We do not leave the above loop until we both have the lock on
        // the node and check the matched status of the locked node.  the
        // lock before (final) read ensures that we will see any write to
        // matched

        unsigned numEdges = std::distance(
            fineGGraph->edge_begin(item, galois::MethodFlag::UNPROTECTED),
            fineGGraph->edge_end(item, galois::MethodFlag::UNPROTECTED));
        // assert(numEdges == std::distance(fineGGraph->edge_begin(item),
        // fineGGraph->edge_end(item)));

        GNode N;
        if (ret != item) {
          // match found
          numEdges += std::distance(
              fineGGraph->edge_begin(ret, galois::MethodFlag::UNPROTECTED),
              fineGGraph->edge_end(ret, galois::MethodFlag::UNPROTECTED));
          // Cautious point
          N = coarseGGraph->createNode(numEdges,
                                       fineGGraph->getData(item).getWeight() +
                                           fineGGraph->getData(ret).getWeight(),
                                       item, ret);
          fineGGraph->getData(item).setMatched();
          fineGGraph->getData(ret).setMatched();
          fineGGraph->getData(item).setParent(N);
          fineGGraph->getData(ret).setParent(N);
        } else {
          // assertAllMatched(item, fineGGraph);
          // Cautious point
          // no match
          if (selfMatch) {
            pc.update(1U);
            N = coarseGGraph->createNode(
                numEdges, fineGGraph->getData(item).getWeight(), item);
            fineGGraph->getData(item).setMatched();
            fineGGraph->getData(item).setParent(N);
          }
        }
      },
      galois::wl<WL>(), galois::no_pushes(), galois::loopname("match"));
}

/*
 * This function is responsible for doing a union find of the edges
 * between matched nodes and populate the edges in the coarser graph
 * node.
 */
void createCoarseEdges(MetisGraph* graph) {
  GGraph* coarseGGraph = graph->getGraph();
  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  assert(fineGGraph != coarseGGraph);

  typedef galois::gstl::Vector<std::pair<GNode, unsigned>> VecTy;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;

  galois::do_all(
      galois::iterate(*coarseGGraph),
      [&](GNode node) {
        //    std::cout << 'p';
        // fineGGraph is read only in this loop, so skip locks
        MetisNode& nodeData =
            coarseGGraph->getData(node, galois::MethodFlag::UNPROTECTED);

        auto& edges = *edgesThreadLocal.getLocal();
        edges.clear();
        for (unsigned x = 0; x < nodeData.numChildren(); ++x) {
          for (auto ii : fineGGraph->edges(nodeData.getChild(x),
                                           galois::MethodFlag::UNPROTECTED)) {
            GNode dst = fineGGraph->getEdgeDst(ii);
            GNode p = fineGGraph->getData(dst, galois::MethodFlag::UNPROTECTED)
                          .getParent();
            edges.emplace_back(p, fineGGraph->getEdgeData(
                                      ii, galois::MethodFlag::UNPROTECTED));
          }
        }

        // slightly faster not ordering by edge weight
        // std::sort(edges.begin(), edges.end(), [] (const std::pair<GNode,
        // unsigned>& lhs, const std::pair<GNode, unsigned>& rhs) { return
        // lhs.first < rhs.first; } );

        // insert edges
        for (auto pp = edges.begin(), ep = edges.end(); pp != ep;) {
          GNode dst    = pp->first;
          unsigned sum = pp->second;
          ++pp;
          if (node != dst) { // no self edges
            while (pp != ep && pp->first == dst) {
              sum += pp->second;
              ++pp;
            }
            coarseGGraph->addMultiEdge(node, dst,
                                       galois::MethodFlag::UNPROTECTED, sum);
          }
        }
        //    assert(e);
        // nodeData.setNumEdges(e);
      },
      galois::steal(), galois::loopname("popedge"));
}

struct HighDegreeIndexer {
  static GGraph* indexgraph;
  unsigned int operator()(const GNode& val) const {
    return indexgraph->getData(val, galois::MethodFlag::UNPROTECTED)
                   .isFailedMatch()
               ? std::numeric_limits<unsigned int>::max()
               : (std::numeric_limits<unsigned int>::max() -
                  ((std::distance(indexgraph->edge_begin(
                                      val, galois::MethodFlag::UNPROTECTED),
                                  indexgraph->edge_end(
                                      val, galois::MethodFlag::UNPROTECTED))) >>
                   2));
  }
};
GGraph* HighDegreeIndexer::indexgraph = 0;

struct LowDegreeIndexer {
  unsigned int operator()(const GNode& val) const {
    unsigned x = std::distance(HighDegreeIndexer::indexgraph->edge_begin(
                                   val, galois::MethodFlag::UNPROTECTED),
                               HighDegreeIndexer::indexgraph->edge_end(
                                   val, galois::MethodFlag::UNPROTECTED));
    return x; // >> 2;
    // int targetlevel = 0;
    // while (x >>= 1) ++targetlevel;
    // return targetlevel;
  }
};

struct WeightIndexer {
  int operator()(const GNode& val) const {
    return HighDegreeIndexer::indexgraph
        ->getData(val, galois::MethodFlag::UNPROTECTED)
        .getWeight();
  }
};

/*unsigned minRuns(unsigned coarsenTo, unsigned size) {
  unsigned num = 0;
  while (coarsenTo < size) {
    ++num;
    size /= 2;
  }
  return num;
}*/

unsigned fixupLoners(GNodeBag& b, GGraph* coarseGGraph, GGraph* fineGGraph) {
  unsigned count = 0;
  auto ii = b.begin(), ee = b.end();
  while (ii != ee) {
    auto i2 = ii;
    ++i2;
    if (i2 != ee) {
      GNode N =
          coarseGGraph->createNode(0,
                                   fineGGraph->getData(*ii).getWeight() +
                                       fineGGraph->getData(*i2).getWeight(),
                                   *ii, *i2);
      fineGGraph->getData(*ii).setMatched();
      fineGGraph->getData(*i2).setMatched();
      fineGGraph->getData(*ii).setParent(N);
      fineGGraph->getData(*i2).setParent(N);
      ++ii;
      ++count;
    } else {
      GNode N = coarseGGraph->createNode(
          0, fineGGraph->getData(*ii).getWeight(), *ii);
      fineGGraph->getData(*ii).setMatched();
      fineGGraph->getData(*ii).setParent(N);
    }
    ++ii;
  }
  return count;
}

unsigned findMatching(MetisGraph* coarseMetisGraph, bool useRM, bool use2Hop,
                      bool verbose) {
  MetisGraph* fineMetisGraph = coarseMetisGraph->getFinerGraph();

  /*
   * Different worklist versions tried, PerSocketChunkFIFO 256 works best with
   * LC_MORPH_graph. Another good type would be Lazy Iter.
   */
  // typedef galois::worklists::ChunkLIFO<64, GNode> WL;
  // typedef
  // galois::worklists::LazyIter<decltype(fineGGraph->local_begin()),false> WL;

  GNodeBag bagOfLoners;
  Pcounter pc;

  bool useOBIM = true;

  typedef galois::worklists::StableIterator<true> WL;
  if (useRM) {
    parallelMatchAndCreateNodes<RMmatch, WL>(coarseMetisGraph, pc, bagOfLoners,
                                             !use2Hop);
  } else {
    // FIXME: use obim for SHEM matching
    typedef galois::worklists::PerSocketChunkLIFO<32> Chunk;
    // typedef galois::worklists::OrderedByIntegerMetric<WeightIndexer, Chunk>
    // pW;
    typedef galois::worklists::OrderedByIntegerMetric<LowDegreeIndexer, Chunk>
        pLD;
    // typedef galois::worklists::OrderedByIntegerMetric<HighDegreeIndexer,
    // Chunk> pHD;

    HighDegreeIndexer::indexgraph = fineMetisGraph->getGraph();
    if (useOBIM)
      parallelMatchAndCreateNodes<HEMmatch, pLD>(coarseMetisGraph, pc,
                                                 bagOfLoners, !use2Hop);
    else
      parallelMatchAndCreateNodes<HEMmatch, WL>(coarseMetisGraph, pc,
                                                bagOfLoners, !use2Hop);
  }
  unsigned c = fixupLoners(bagOfLoners, coarseMetisGraph->getGraph(),
                           fineMetisGraph->getGraph());
  if (verbose && c)
    std::cout << "\n\tLone Matches " << c;
  if (use2Hop) {
    typedef galois::worklists::PerSocketChunkLIFO<32> Chunk;
    // typedef galois::worklists::OrderedByIntegerMetric<WeightIndexer, Chunk>
    // pW;
    typedef galois::worklists::OrderedByIntegerMetric<LowDegreeIndexer, Chunk>
        pLD;
    // typedef galois::worklists::OrderedByIntegerMetric<HighDegreeIndexer,
    // Chunk> pHD;

    HighDegreeIndexer::indexgraph = fineMetisGraph->getGraph();
    Pcounter pc2;
    if (useOBIM)
      parallelMatchAndCreateNodes<TwoHopMatcher<HEMmatch>, pLD>(
          coarseMetisGraph, pc2, bagOfLoners, true);
    else
      parallelMatchAndCreateNodes<TwoHopMatcher<HEMmatch>, WL>(
          coarseMetisGraph, pc2, bagOfLoners, true);
    return pc2.reduce();
  }
  return pc.reduce();
}

MetisGraph* coarsenOnce(MetisGraph* fineMetisGraph, unsigned& rem, bool useRM,
                        bool with2Hop, bool verbose) {
  MetisGraph* coarseMetisGraph = new MetisGraph(fineMetisGraph);
  galois::Timer t, t2;
  if (verbose)
    t.start();
  rem = findMatching(coarseMetisGraph, useRM, with2Hop, verbose);
  if (verbose) {
    t.stop();
    std::cout << "\n\tTime Matching " << t.get() << "\n";
    t2.start();
  }
  createCoarseEdges(coarseMetisGraph);
  if (verbose) {
    t2.stop();
    std::cout << "\tTime Creating " << t2.get() << "\n";
  }
  return coarseMetisGraph;
}

} // namespace

MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo,
                    bool verbose) {
  MetisGraph* coarseGraph = fineMetisGraph;
  unsigned size           = std::distance(fineMetisGraph->getGraph()->begin(),
                                fineMetisGraph->getGraph()->end());
  unsigned iterNum        = 0;
  bool with2Hop           = false;
  unsigned stat           = 0;
  while (true) { // overflow
    if (verbose) {
      std::cout << "Coarsening " << iterNum << "\t";
      stat = graphStat(*coarseGraph->getGraph());
    }
    unsigned rem     = 0;
    coarseGraph      = coarsenOnce(coarseGraph, rem, false, with2Hop, verbose);
    unsigned newSize = size / 2 + rem / 2;
    if (verbose) {
      std::cout << "\tTO\t";
      unsigned stat2 = graphStat(*coarseGraph->getGraph());
      std::cout << "\n\tRatio " << (double)stat2 / (double)stat << " REM "
                << rem << " new size " << newSize << "\n";
    }

    if (size * 3 < newSize * 4) {
      with2Hop = true;
      if (verbose)
        std::cout << "** Enabling 2 hop matching\n";
    } else {
      with2Hop = false;
    }

    size = newSize;
    if (newSize * 4 < coarsenTo) { // be more exact near the end
      size = std::distance(coarseGraph->getGraph()->begin(),
                           coarseGraph->getGraph()->end());
      if (size < coarsenTo)
        break;
    }
    ++iterNum;
  }

  return coarseGraph;
}
