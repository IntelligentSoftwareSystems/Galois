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

#ifndef _KRUSKAL_ODG_H_
#define KRUSKAL_ODG_H_

#include <tr1/unordered_set>

#include "galois/Atomic.h"
#include "galois/Reduction.h"

#include "kruskal.h"

// TODO
//
// ===== Build the ODG ====
// for each edge ke
//    add ke to adj set of nodes kn1 & kn2 of ke
//    create a ODG node de for ke
// end
//
// for each node kn
//    -- create clique between edges
//    for i in 1 : adjset.size
//      for j in i : adjset.size
//        add edge between dag nodes adj[i] & adj[j]
//      end
//    end
// end
//
// === adding edge in ODG ===
// adding edge betwen dag nodes dn1 and dn2
// let ni = min (dn1, dn2) and nj = max (dn1, dn2)
// where comparison is based on kruskal edges corresponding to dn1 and dn2
// then
// add nj to adj list of ni
// increment indeg counter of nj
//
// ===== Run kruskal =======
// for each node in ODG
//    if indeg == 0 then add to workset
// end
//
// for each node n in workset
//    run kruskal on edge corresponding to n
//    for each neighbor m of n in ODG
//      create clique between m and other neighbors
//    end
//
//    for each neighbor m of n
//      decrement indeg
//      if indeg == 0 then add to thread local list
//    end
//
//    for each node in thread local list
//      add node to workset
//    end
// end
//

class KruskalODG : public Kruskal {

private:
  struct ODGnode;

  static const unsigned CHUNK_SIZE = 16;
  typedef galois::runtime::worklists::ChunkFIFO<CHUNK_SIZE, ODGnode*> WLTy;

  typedef galois::GAccumulator<size_t> Accumulator;
  typedef galois::runtime::PerCPU<std::vector<ODGnode*>> VecPerThrd;

  struct ODGnode {

    typedef std::tr1::unordered_set<ODGnode*> OutSet;

    size_t id;
    KEdge* activeElem;
    galois::GAtomic<unsigned> indeg;

  private:
    OutSet outNeigh;
    PaddedLock<true> lock;

  private:
    bool addOutNeigh(ODGnode* const dst) {
      assert(dst != NULL);
      bool retval = false;

      lock.lock();
      {
        if (outNeigh.find(dst) == outNeigh.end()) {

          outNeigh.insert(dst);
          retval = true;

        } else {
          retval = false;
        }
      }
      lock.unlock();

      return retval;
    }

  public:
    ODGnode(size_t id, KEdge* edge) : id(id), activeElem(edge), indeg(0) {}

    int compare(const ODGnode& that) const {
      int cmp = this->activeElem->weight - that.activeElem->weight;

      if (cmp == 0) {
        cmp = this->id - that.id;
      }

      assert((this != &that) && (cmp != 0));

      return cmp;
    }

    bool operator<(const ODGnode& that) const { return (compare(that) < 0); }

    //! adds a directed edge from smaller to bigger
    static void addEdge(ODGnode& left, ODGnode& right) {
      ODGnode* src;
      ODGnode* dst;

      if (left.compare(right) <= 0) {
        src = &left;
        dst = &right;

      } else {
        src = &right;
        dst = &left;
      }

      // add dst to outgoing list of src
      // increment indeg counter of dst
      if (src->addOutNeigh(dst)) {
        ++(dst->indeg);
      }
    }

    OutSet::iterator neighbor_begin() { return outNeigh.begin(); }

    OutSet::const_iterator neighbor_begin() const { return outNeigh.begin(); }

    OutSet::iterator neighbor_end() { return outNeigh.end(); }

    OutSet::const_iterator neighbor_end() const { return outNeigh.end(); }
  };

  struct LoopBody {

    VecPerThrd& addListPerThrd;
    Accumulator& mstSum;
    Accumulator& numIter;

    LoopBody(VecPerThrd& addListPerThrd, Accumulator& mstSum,
             Accumulator& numIter)
        : addListPerThrd(addListPerThrd), mstSum(mstSum), numIter(numIter) {}

    template <typename ContextTy>
    void operator()(ODGnode* src, ContextTy& lwl) {

      // std::cout << "Running iteration: " << numIter.get () << std::endl;

      KEdge* edge = src->activeElem;

      if (edge->contract()) {
        mstSum.get() += edge->weight;
      }

      // TODO: the following may be moved inside if statement above(contract
      // edge)
      for (ODGnode::OutSet::iterator i  = src->neighbor_begin(),
                                     ei = src->neighbor_end();
           i != ei; ++i) {

        ODGnode::OutSet::iterator j = i, ej = ei;
        ++j; // j = i + 1;
        for (; j != ej; ++j) {
          if ((*i) != (*j)) {
            ODGnode::addEdge(*(*i), *(*j));
          }
        }
      }

      for (ODGnode::OutSet::iterator i  = src->neighbor_begin(),
                                     ei = src->neighbor_end();
           i != ei; ++i) {

        size_t deg = --((*i)->indeg);
        if (deg == 0) {
          addListPerThrd.get().push_back(*i);
        }
      }

      // XXX: it may be the case that we add only one item in kruskal

      for (std::vector<ODGnode*>::const_iterator
               i  = addListPerThrd.get().begin(),
               ei = addListPerThrd.get().end();
           i != ei; ++i) {
        lwl.push(*i);
      }

      addListPerThrd.get().clear();

      ++(numIter.get());
    }
  };

protected:
  virtual const std::string getVersion() const { return "ODG Kruskal"; }

  virtual void runMST(std::vector<KNode*>& nodes, VecKEdge_ty& edges,
                      size_t& mstWeight, size_t& totalIter) {

    std::vector<std::vector<size_t>> nodeAdjLists(nodes.size());

    std::vector<ODGnode> odgNodes;

    size_t idCntr = 0;

    for (std::set<KEdge>::iterator e = edges.begin(), ee = edges.end(); e != ee;
         ++e) {

      ODGnode nd(idCntr, const_cast<KEdge*>(&(*e)));
      ++idCntr;
      odgNodes.push_back(nd);

      nodeAdjLists[e->src->id].push_back(nd.id);
      nodeAdjLists[e->dst->id].push_back(nd.id);
    }

    for (size_t a = 0; a < nodeAdjLists.size(); ++a) {

      // create a directed clique
      for (size_t i = 0; i < nodeAdjLists[a].size(); ++i) {
        for (size_t j = i + 1; j < nodeAdjLists[a].size(); ++j) {

          size_t idx1 = nodeAdjLists[a][i];
          size_t idx2 = nodeAdjLists[a][j];

          ODGnode::addEdge(odgNodes[idx1], odgNodes[idx2]);
        }
      }
    }

    // TODO: use for_each filters here

    std::vector<ODGnode*> initWL;
    for (std::vector<ODGnode>::iterator i  = odgNodes.begin(),
                                        ei = odgNodes.end();
         i != ei; ++i) {

      if (i->indeg == 0) {
        initWL.push_back(&(*i));
      }
    }

    VecPerThrd addListPerThrd;
    Accumulator mstSum;
    Accumulator numIter;

    galois::for_each<WLTy>(initWL.begin(), initWL.end(),
                           LoopBody(addListPerThrd, mstSum, numIter));

    mstWeight = mstSum.get();
    totalIter = numIter.get();
  }
};

#endif //  KRUSKAL_ODG_H_
