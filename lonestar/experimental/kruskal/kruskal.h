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

#ifndef _KRUSKAL_H_
#define _KRUSKAL_H_

#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <set>

#include <cstdlib>
#include <cstdio>

#include <boost/iterator/counting_iterator.hpp>

#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/Graph/FileGraph.h"
#include "galois/Graph/LCGraph.h"
#include "galois/runtime/WorkList.h"
#include "galois/runtime/Profile.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "kruskalData.h"

namespace cll = llvm::cl;

static const char* const name = "Kruskal MST";
static const char* const desc =
    "Computes minimum weight spanning tree of an undirected graph";
static const char* const url = "kruskal";

static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);

template <typename KNode_tp>
class Kruskal {

public:
  typedef Kruskal<KNode_tp> Base_ty;

  typedef unsigned Weight_ty;
  typedef std::vector<KNode_tp*> VecKNode_ty;
  typedef std::vector<KEdge<KNode_tp>*> VecKEdge_ty;

protected:
  typedef std::set<KEdge<KNode_tp>, typename KEdge<KNode_tp>::NodeIDcomparator>
      Edges_ty;
  // typedef std::vector<KEdge<KNode_tp> > Edges_ty;

  virtual const std::string getVersion() const = 0;

  //! doesn't do anything by default. Sub-classes may choose to override
  //! in order to to specific initialization
  virtual void initRemaining(VecKNode_ty& nodes, VecKEdge_ty& edges){};

  virtual void runMST(VecKNode_ty& nodes, VecKEdge_ty& edges, size_t& mstWeight,
                      size_t& totalIter) = 0;

  void readGraph(const std::string& filename, VecKNode_ty& nodes,
                 Edges_ty& edges) {

    typedef galois::graphs::LC_CSR_Graph<unsigned, unsigned> InGraph;
    typedef InGraph::GraphNode InGNode;

    InGraph ingraph;
    galois::graphs::readGraph(ingraph, filename);

    // numbering nodes 0..N-1, where N is number of nodes
    // in the graph
    unsigned idCntr = 0;
    for (InGraph::iterator n = ingraph.begin(), endn = ingraph.end(); n != endn;
         ++n) {
      ingraph.getData(*n, galois::MethodFlag::UNPROTECTED) = idCntr;
      ++idCntr;
    }

    nodes.resize(ingraph.size(), NULL);
    // edges.reserve (ingraph.sizeEdges ());

    size_t numEdges = 0;

    for (InGraph::iterator n = ingraph.begin(), endn = ingraph.end(); n != endn;
         ++n) {

      InGNode src = ingraph.getData(*n, galois::MethodFlag::UNPROTECTED);

      if (nodes[src] == NULL) {
        nodes[src] = new KNode_tp(src);
      }

      for (InGraph::edge_iterator
               e    = ingraph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
               ende = ingraph.edge_end(src, galois::MethodFlag::UNPROTECTED);
           e != ende; ++e) {

        InGNode dst = ingraph.getEdgeDst(*e);

        if (nodes[dst] == NULL) {
          nodes[dst] = new KNode_tp(dst);
        }

        if (src != dst) {
          KEdge<KNode_tp> ke(nodes[src], nodes[dst], ingraph.getEdgeData(*e));

          // edges.push_back (ke);
          std::pair<typename Edges_ty::iterator, bool> res = edges.insert(ke);

          if (res.second) {
            ++numEdges;
          }

        } else {
          std::fprintf(stderr, "Warning: Ignoring self edge (%d, %d, %d)\n",
                       src, dst, ingraph.getEdgeData(*e));
        }
      }
    }

    std::cout << "Graph read with nodes=" << ingraph.size()
              << ", edges=" << numEdges << std::endl;
  }

  virtual void readPBBSfile(const std::string& filename, VecKNode_ty& nodes,
                            Edges_ty& edges) {

    typedef unsigned NodeData;
    typedef float EdgeData;

    static const unsigned WEIGHT_SCALE = 10000;

    nodes.clear();
    edges.clear();

    std::cout << "Reading input from: " << filename << std::endl;

    // std::ifstream inFile (filename.c_str ());
    FILE* inFile = fopen(filename.c_str(), "r");

    // std::string header;
    char header[128];

    // inFile >> header;
    fscanf(inFile, "%s", header);

    // inFile.seekg (0, std::ios::beg);

    size_t numEdges = 0;

    // while (!inFile.eof ()) {
    while (!feof(inFile)) {
      NodeData srcIdx;
      NodeData dstIdx;
      EdgeData w;

      // inFile >> srcIdx;
      // inFile >> dstIdx;
      // inFile >> w;

      fscanf(inFile, "%d", &srcIdx);
      fscanf(inFile, "%d", &dstIdx);
      fscanf(inFile, "%g", &w);

      if (nodes.size() <= srcIdx) {
        nodes.resize((srcIdx + 1), NULL);
      }

      if (nodes.size() <= dstIdx) {
        nodes.resize((dstIdx + 1), NULL);
      }

      if (nodes[srcIdx] == NULL) {
        nodes[srcIdx] = new KNode_tp(srcIdx);
      }

      if (nodes[dstIdx] == NULL) {
        nodes[dstIdx] = new KNode_tp(dstIdx);
      }

      Weight_ty integ_wt = (WEIGHT_SCALE * w);

      if (srcIdx != dstIdx) {

        KEdge<KNode_tp> ke(nodes[srcIdx], nodes[dstIdx], integ_wt);
        std::pair<typename Edges_ty::iterator, bool> res = edges.insert(ke);
        // edges.push_back (ke);
        if (res.second) {
          ++numEdges;
        }

      } else {
        std::fprintf(stderr, "Warning: Ignoring self edge (%d, %d, %d)\n",
                     srcIdx, dstIdx, integ_wt);
      }
    }
    // inFile.close ();
    fclose(inFile);

    std::cout << "PBBS graph read with nodes = " << nodes.size()
              << ", edges = " << numEdges << std::endl;
  }

public:
  virtual void run(int argc, char* argv[]) {
    galois::StatManager stat;
    LonestarStart(argc, argv, name, desc, url);

    // TODO
    // read the graph from file into a FileGraph
    // create nodes and edges
    // compute a set of edges
    // run kruskal, which should return mst weight as int
    // verify

    VecKNode_ty nodes;
    Edges_ty edges;

    size_t mstWeight = 0;
    size_t totalIter = 0;

    readGraph(filename, nodes, edges);
    // readPBBSfile (filename, nodes, edges);
    //
    VecKEdge_ty edgesVec;
    edgesVec.reserve(edges.size());

    for (typename Edges_ty::iterator i = edges.begin(), endi = edges.end();
         i != endi; ++i) {
      // edgesVec.push_back (&(*i));
      edgesVec.push_back(const_cast<KEdge<KNode_tp>*>(&(*i)));
    }

    initRemaining(nodes, edgesVec);

    galois::StatTimer t("Time taken by runMST: ");

    t.start();
    runMST(nodes, edgesVec, mstWeight, totalIter);
    t.stop();

    printResults(mstWeight, totalIter);

    if (!skipVerify) {
      verify(nodes, edges, mstWeight);
    }

    freeVecPtr(nodes);
  }

private:
  void printResults(const size_t mstSum, const size_t iter) const {
    std::cout << getVersion() << ", MST sum=" << mstSum
              << ", iterations=" << iter << std::endl;
  }

  template <typename T>
  static void freeVecPtr(std::vector<T*>& vec) {
    for (typename std::vector<T*>::iterator i = vec.begin(), ei = vec.end();
         i != ei; ++i) {

      delete *i;
      *i = NULL;
    }
  }

  struct PrimNode {
    typedef std::vector<PrimNode*> VecPNode_ty;
    typedef std::vector<Weight_ty> VecWeight_ty;
    typedef boost::counting_iterator<unsigned> Adj_iterator_ty;

    unsigned id;
    size_t weight;
    bool inMST;
    VecPNode_ty adj;
    VecWeight_ty adjWts;

    PrimNode(unsigned id)
        : id(id), weight(std::numeric_limits<Weight_ty>::max()), inMST(false) {}

    void addEdge(PrimNode* pn, Weight_ty w) {
      assert(pn != NULL);
      assert(std::find(adj.begin(), adj.end(), pn) == adj.end());

      adj.push_back(pn);
      adjWts.push_back(w);
      assert(adj.size() == adjWts.size());
    }

    Adj_iterator_ty adj_begin() const { return Adj_iterator_ty(0); }

    Adj_iterator_ty adj_end() const { return Adj_iterator_ty(adj.size()); }

    PrimNode* getDst(Adj_iterator_ty i) const { return adj[*i]; }

    Weight_ty getWeight(Adj_iterator_ty i) const { return adjWts[*i]; }
  };

  struct PrimUpdate {
    PrimNode* dst;
    size_t weight;

    PrimUpdate(PrimNode* dst, size_t weight) : dst(dst), weight(weight) {
      assert(dst != NULL);
    }

    bool operator<(const PrimUpdate& that) const {
      if (this->weight == that.weight) {
        return (this->dst->id < that.dst->id);

      } else {
        return (this->weight < that.weight);
      }
    }
  };

  static unsigned getID(KNode_tp* nd) { return nd->id; }

  size_t runPrim(const VecKNode_ty& nodes, const Edges_ty& edges) const {

    std::vector<PrimNode*> primNodes(nodes.size(), NULL);

    for (typename VecKNode_ty::const_iterator n    = nodes.begin(),
                                              endn = nodes.end();
         n != endn; ++n) {
      unsigned index   = getID(*n);
      primNodes[index] = new PrimNode(index);
    }

    for (typename Edges_ty::const_iterator e    = edges.begin(),
                                           ende = edges.end();
         e != ende; ++e) {

      unsigned srcIdx = getID(e->src);
      unsigned dstIdx = getID(e->dst);

      assert(primNodes[srcIdx] != NULL);
      assert(primNodes[dstIdx] != NULL);

      // add undirected edge
      primNodes[srcIdx]->addEdge(primNodes[dstIdx], e->weight);
      primNodes[dstIdx]->addEdge(primNodes[srcIdx], e->weight);
    }

    std::set<PrimUpdate> workset;

    PrimNode* root = primNodes[0];
    PrimUpdate upd(root, 0);
    workset.insert(upd);

    size_t iter   = 0;
    size_t mstSum = 0;

    while (!workset.empty()) {
      ++iter;
      PrimUpdate upd = *(workset.begin());
      workset.erase(workset.begin());

      PrimNode& src = *(upd.dst);

      if (!src.inMST) {
        src.inMST  = true;
        src.weight = upd.weight;

        mstSum += upd.weight;

        for (typename PrimNode::Adj_iterator_ty i    = src.adj_begin(),
                                                endi = src.adj_end();
             i != endi; ++i) {

          PrimNode& dst = *(src.getDst(i));
          Weight_ty wt  = src.getWeight(i);

          if (!dst.inMST) {
            PrimUpdate addUpd(&dst, wt);
            workset.insert(addUpd);
          }
        }
      } // end if;
    }

    std::cout << "Number of iterations taken by Prim = " << iter << std::endl;

    freeVecPtr(primNodes);
    return mstSum;
  }

  bool verify(const VecKNode_ty& nodes, const Edges_ty& edges,
              const size_t kruskalSum) const {
    galois::StatTimer pt("Prim's Time:");
    pt.start();
    size_t primSum = runPrim(nodes, edges);
    pt.stop();

    if (primSum != kruskalSum) {
      std::cerr << "ERROR. Incorrect MST weight=" << kruskalSum
                << ", correct weight from Prim is=" << primSum << std::endl;
      abort();

    } else {
      std::cout << "OK. Correct MST weight=" << kruskalSum << std::endl;
    }

    return false;
  }
};

#endif // _KRUSKAL_H_
