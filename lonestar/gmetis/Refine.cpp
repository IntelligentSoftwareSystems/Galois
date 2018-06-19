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

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "Metis.h"
#include <set>
#include <iostream>

namespace {

bool isBoundary(GGraph& g, GNode n) {
  unsigned int nPart = g.getData(n).getPart();
  for (auto ii : g.edges(n))
    if (g.getData(g.getEdgeDst(ii)).getPart() != nPart)
      return true;
  return false;
}

// This is only used on the terminal graph (find graph)
void findBoundary(GNodeBag& bag, GGraph& cg) {

  galois::do_all(galois::iterate(cg),
                 [&](GNode n) {
                   auto& cn = cg.getData(n, galois::MethodFlag::UNPROTECTED);
                   if (cn.getmaybeBoundary())
                     cn.setmaybeBoundary(isBoundary(cg, n));
                   if (cn.getmaybeBoundary())
                     bag.push(n);
                 },
                 galois::loopname("findBoundary"));
}

// this is used on the coarse graph to project to the fine graph
void findBoundaryAndProject(GNodeBag& bag, GGraph& cg, GGraph& fg) {
  galois::do_all(galois::iterate(cg),
                 [&](GNode n) {
                   auto& cn = cg.getData(n, galois::MethodFlag::UNPROTECTED);
                   if (cn.getmaybeBoundary())
                     cn.setmaybeBoundary(isBoundary(cg, n));

                   // project part and maybe boundary
                   // unsigned part = cn.getPart();
                   for (unsigned x = 0; x < cn.numChildren(); ++x) {
                     fg.getData(cn.getChild(x), galois::MethodFlag::UNPROTECTED)
                         .initRefine(cn.getPart(), cn.getmaybeBoundary());
                   }
                   if (cn.getmaybeBoundary())
                     bag.push(n);
                 },
                 galois::loopname("findBoundaryAndProject"));
}

template <bool balance>
void refine_BKL2(unsigned minSize, unsigned maxSize, GGraph& cg, GGraph* fg,
                 std::vector<partInfo>& parts) {

  auto gainIndexer = [&cg](GNode n) -> int {
    int retval              = 0;
    galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    unsigned int nPart      = cg.getData(n, flag).getPart();
    for (auto ii = cg.edge_begin(n, flag), ee = cg.edge_end(n); ii != ee;
         ++ii) {
      GNode neigh = cg.getEdgeDst(ii);
      if (cg.getData(neigh, flag).getPart() == nPart)
        retval -= cg.getEdgeData(ii, flag);
      else
        retval += cg.getEdgeData(ii, flag);
    }
    return -retval / 16;
  };

  typedef galois::worklists::PerSocketChunkFIFO<8> Chunk;
  typedef galois::worklists::OrderedByIntegerMetric<decltype(gainIndexer),
                                                    Chunk, 10>
      pG;

  GNodeBag boundary;

  if (fg)
    findBoundaryAndProject(boundary, cg, *fg);
  else
    findBoundary(boundary, cg);

  //! [Example Per-Thread-Storage Declaration]
  typedef galois::gstl::Vector<unsigned> VecTy;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  //! [Example Per-Thread-Storage Declaration]

  //! [Example Per-Thread-Storage Usage]
  // Find the partition n is most connected to
  auto pickPartitionEC = [&](GNode n, auto& cnx) -> unsigned {
    auto& edges = *edgesThreadLocal.getLocal();
    edges.clear();
    edges.resize(parts.size(), 0);
    unsigned P = cg.getData(n).getPart();
    for (auto ii : cg.edges(n)) {
      GNode neigh = cg.getEdgeDst(ii);
      auto& nd    = cg.getData(neigh);
      if (parts[nd.getPart()].partWeight < maxSize || nd.getPart() == P)
        edges[nd.getPart()] += cg.getEdgeData(ii);
    }
    return std::distance(edges.begin(),
                         std::max_element(edges.begin(), edges.end()));
  };
  //! [Example Per-Thread-Storage Usage]

  // Find the smallest partition n is connected to
  auto pickPartitionMP = [&](GNode n, auto& cnx) -> unsigned {
    unsigned P  = cg.getData(n).getPart();
    unsigned W  = parts[P].partWeight;
    auto& edges = *edgesThreadLocal.getLocal();
    edges.clear();
    edges.resize(parts.size(), ~0);
    edges[P] = W;
    W        = (double)W * 0.9;
    for (auto ii : cg.edges(n)) {
      GNode neigh = cg.getEdgeDst(ii);
      auto& nd    = cg.getData(neigh);
      if (parts[nd.getPart()].partWeight < W)
        edges[nd.getPart()] = parts[nd.getPart()].partWeight;
    }
    return std::distance(edges.begin(),
                         std::min_element(edges.begin(), edges.end()));
  };

  galois::for_each(
      galois::iterate(boundary),
      [&](GNode n, auto& cnx) {
        auto& nd         = cg.getData(n);
        unsigned curpart = nd.getPart();
        unsigned newpart =
            balance ? pickPartitionMP(n, cnx) : pickPartitionEC(n, cnx);
        if (parts[curpart].partWeight < minSize)
          return;
        if (curpart != newpart) {
          nd.setPart(newpart);
          __sync_fetch_and_sub(&parts[curpart].partWeight, nd.getWeight());
          __sync_fetch_and_add(&parts[newpart].partWeight, nd.getWeight());
          for (auto ii : cg.edges(n)) {
            GNode neigh = cg.getEdgeDst(ii);
            auto& ned   = cg.getData(neigh);
            if (ned.getPart() != newpart && !ned.getmaybeBoundary()) {
              ned.setmaybeBoundary(true);
              if (fg)
                for (unsigned x = 0; x < ned.numChildren(); ++x)
                  fg->getData(ned.getChild(x), galois::MethodFlag::UNPROTECTED)
                      .setmaybeBoundary(true);
            }
            // if (ned.getPart() != newpart)
            // cnx.push(neigh);
          }
          if (fg)
            for (unsigned x = 0; x < nd.numChildren(); ++x)
              fg->getData(nd.getChild(x), galois::MethodFlag::UNPROTECTED)
                  .setPart(newpart);
        }
      },
      galois::loopname("refine"), galois::wl<pG>(gainIndexer));
}

void projectPart(MetisGraph* Graph, std::vector<partInfo>& parts) {
  GGraph* fineGraph   = Graph->getFinerGraph()->getGraph();
  GGraph* coarseGraph = Graph->getGraph();

  galois::do_all(galois::iterate(*coarseGraph),
                 [&](GNode n) {
                   auto& cn      = coarseGraph->getData(n);
                   unsigned part = cn.getPart();
                   for (unsigned x = 0; x < cn.numChildren(); ++x) {
                     fineGraph->getData(cn.getChild(x)).setPart(part);
                   }
                 },
                 galois::loopname("project"));
}

int gain(GGraph& g, GNode n) {
  int retval         = 0;
  unsigned int nPart = g.getData(n).getPart();
  for (auto ii : g.edges(n)) {
    GNode neigh = g.getEdgeDst(ii);
    if (g.getData(neigh).getPart() == nPart)
      retval -= g.getEdgeData(ii);
    else
      retval += g.getEdgeData(ii);
  }
  return retval;
}

void parallelBoundary(GNodeBag& bag, GGraph& graph) {
  galois::do_all(galois::iterate(graph),
                 [&](GNode n) {
                   if (gain(graph, n) > 0)
                     bag.push(n);
                 },
                 galois::loopname("Get-Boundary"));
}

void refineOneByOne(GGraph& g, std::vector<partInfo>& parts) {
  std::vector<GNode> boundary;
  unsigned int meanWeight = 0;
  for (unsigned int i = 0; i < parts.size(); i++)
    meanWeight += parts[i].partWeight;
  meanWeight /= parts.size();

  GNodeBag boundaryBag;
  parallelBoundary(boundaryBag, g);

  for (auto ii = boundaryBag.begin(), ie = boundaryBag.end(); ii != ie; ii++) {
    GNode n        = (*ii);
    unsigned nPart = g.getData(n).getPart();
    int part[parts.size()];
    for (unsigned int i = 0; i < parts.size(); i++)
      part[i] = 0;
    for (auto ii : g.edges(n)) {
      GNode neigh = g.getEdgeDst(ii);
      part[g.getData(neigh).getPart()] += g.getEdgeData(ii);
    }
    int t          = part[nPart];
    unsigned int p = nPart;
    for (unsigned int i = 0; i < parts.size(); i++)
      if (i != nPart && part[i] > t &&
          parts[nPart].partWeight > parts[i].partWeight * (98) / (100) &&
          parts[nPart].partWeight > meanWeight * 98 / 100) {
        t = part[i];
        p = i;
      }
    if (p != nPart) {
      g.getData(n).setPart(p);
      parts[p].partWeight += g.getData(n).getWeight();
      parts[nPart].partWeight -= g.getData(n).getWeight();
    }
  }
}

void refine_BKL(GGraph& g, std::vector<partInfo>& parts) {
  std::set<GNode> boundary;

  // find boundary nodes with positive gain
  GNodeBag boundaryBag;
  parallelBoundary(boundaryBag, g);

  for (auto ii = boundaryBag.begin(), ie = boundaryBag.end(); ii != ie; ii++) {
    boundary.insert(*ii);
  }

  // refine by swapping with a neighbor high-gain node
  while (!boundary.empty()) {
    GNode n = *boundary.begin();
    boundary.erase(boundary.begin());
    unsigned nPart = g.getData(n).getPart();
    for (auto ii : g.edges(n)) {
      GNode neigh        = g.getEdgeDst(ii);
      unsigned neighPart = g.getData(neigh).getPart();
      if (neighPart != nPart && boundary.count(neigh) && gain(g, n) > 0 &&
          gain(g, neigh) > 0) {
        unsigned nWeight     = g.getData(n).getWeight();
        unsigned neighWeight = g.getData(neigh).getWeight();
        // swap
        g.getData(n).setPart(neighPart);
        g.getData(neigh).setPart(nPart);
        // update partinfo
        parts[neighPart].partWeight += nWeight;
        parts[neighPart].partWeight -= neighWeight;
        parts[nPart].partWeight += neighWeight;
        parts[nPart].partWeight -= nWeight;
        // remove nodes
        boundary.erase(neigh);
        break;
      }
    }
  }
}

/*double ratiocut(int nbClust, int* degree, int* card)
{
  double res=0;
  for (int i=0; i<nbClust;i++)
    res += (double)(degree[i])/(double)(card[i]);

  return res;
}*/

void GraclusRefining(GGraph* graph, int nbParti, int nbIter) {
  nbIter = std::min(15, nbIter);
  std::vector<double> Dist(nbParti);
  std::vector<int> card(nbParti);
  std::vector<int> degreeIn(nbParti);

  using Accum = galois::GAccumulator<size_t>;
  std::vector<Accum> cardAccum(nbParti);
  std::vector<Accum> degreeInAccum(nbParti);

  for (int j = 0; j < nbIter; j++) {

    GGraph& g = *graph;
    galois::do_all(
        galois::iterate(g),
        [&](GNode n) {
          unsigned int clust =
              g.getData(n, galois::MethodFlag::UNPROTECTED).getPart();
          int degreet = 0;

          g.getData(n, galois::MethodFlag::UNPROTECTED).OldPartCpyNew();

          for (auto ii : g.edges(n, galois::MethodFlag::UNPROTECTED))
            if (g.getData(g.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED)
                    .getPart() == clust)
              degreet +=
                  (int)g.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);

          cardAccum[clust] +=
              g.getData(n, galois::MethodFlag::UNPROTECTED).getWeight();
          degreeInAccum[clust] += degreet;
        },
        galois::loopname("compute dists"));

    for (int i = 0; i < nbParti; i++) {
      card[i] = cardAccum[i].reduce();
      cardAccum[i].reset();

      degreeIn[i] = degreeInAccum[i].reduce();
      degreeInAccum[i].reset();

      Dist[i] = (card[i] != 0) ? (double)(degreeIn[i] + card[i]) /
                                     ((double)card[i] * card[i])
                               : 0;
    }

    galois::do_all(
        galois::iterate(g),
        [&](GNode n) {
          double dmin   = std::numeric_limits<double>::min();
          int partition = -1;
          galois::gstl::Map<int, int> degreein;
          degreein[g.getData(n, galois::MethodFlag::UNPROTECTED)
                       .getOldPart()] += 1;
          for (auto ii : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
            int nclust =
                g.getData(g.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED)
                    .getOldPart();
            degreein[nclust] +=
                (int)g.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);
          }

          for (auto clust = degreein.begin(), ee = degreein.end(); clust != ee;
               ++clust) {
            // the distance between the cluster clust and the noden is :
            double d = Dist[clust->first] - (2.0 * (double)clust->second /
                                             (double)card[clust->first]);
            if (d < dmin || partition == -1) {
              dmin      = d;
              partition = clust->first;
            }
          }
          g.getData(n, galois::MethodFlag::UNPROTECTED).setPart(partition);
        },
        galois::loopname("make moves"));
  }
  /*  std::cout<<ratiocut(nbParti, degreeIn, card)<< '\n';
  for (int i=0; i<nbParti; i++)
    std::cout<<card[i]<< ' ';
  std::cout<<std::endl;*/
}

} // namespace

void refine(MetisGraph* coarseGraph, std::vector<partInfo>& parts,
            unsigned minSize, unsigned maxSize, refinementMode refM,
            bool verbose) {
  MetisGraph* tGraph = coarseGraph;
  int nbIter         = 1;
  if (refM == GRACLUS) {
    while ((tGraph = tGraph->getFinerGraph()))
      nbIter *= 2;
    nbIter /= 4;
  }
  do {
    MetisGraph* fineGraph = coarseGraph->getFinerGraph();
    bool doProject        = true;
    if (verbose) {
      std::cout << "Cut " << computeCut(*coarseGraph->getGraph())
                << " Weights ";
      printPartStats(parts);
      std::cout << "\n";
    }
    // refine nparts times
    switch (refM) {
    case BKL2:
      refine_BKL2<false>(minSize, maxSize, *coarseGraph->getGraph(),
                         fineGraph ? fineGraph->getGraph() : nullptr, parts);
      doProject = false;
      break;
    case BKL:
      refine_BKL(*coarseGraph->getGraph(), parts);
      break;
    case ROBO:
      refineOneByOne(*coarseGraph->getGraph(), parts);
      break;
    case GRACLUS:
      GraclusRefining(coarseGraph->getGraph(), parts.size(), nbIter);
      nbIter = (nbIter + 1) / 2;
      break;
    default:
      abort();
    }
    // project up
    if (fineGraph && doProject) {
      projectPart(coarseGraph, parts);
    }
  } while ((coarseGraph = coarseGraph->getFinerGraph()));
}

/*
void balance(MetisGraph* coarseGraph, std::vector<partInfo>& parts, unsigned
meanSize) { MetisGraph* fineGraph = coarseGraph->getFinerGraph();
    refine_BKL2<true>(meanSize, *coarseGraph->getGraph(), fineGraph ?
fineGraph->getGraph() : nullptr, parts);
}
*/
