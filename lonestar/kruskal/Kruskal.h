/** Kruskal MST -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @section Description
 *
 * Kruskal MST.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef _KRUSKAL_H_ 
#define _KRUSKAL_H_ 

#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <set>
#include <unordered_set>

#include <cstdlib>
#include <cstdio>

#include <boost/iterator/counting_iterator.hpp>

#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Timer.h"
#include "Galois/Galois.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/WorkList/WorkList.h"
#include "Galois/Runtime/Sampling.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;

static const char* const name = "Kruskal's Minimum Spanning Tree Algorithm ";
static const char* const desc = "Computes minimum weight spanning tree of an undirected graph";
static const char* const url = "mst";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

static cll::opt<unsigned> numPages (
    "preAlloc",
    cll::desc ("number of pages (per thread) to pre-allocate from OS for Galois allocators"),
    cll::init (32));

namespace kruskal {

typedef galois::GAccumulator<size_t> Accumulator;

typedef unsigned Weight_ty;
typedef std::vector<int> VecRep;
struct Edge;
typedef std::vector<Edge> VecEdge;

struct InEdge {
  int src;
  int dst;
  Weight_ty weight;

  InEdge (int _src, int _dst, Weight_ty _weight)
    : src (_src), dst (_dst), weight (_weight) 
  {
    if (src == dst) {
      fprintf (stderr, "Self edges not allowed\n");
      abort ();
    }

    // nodes in an edge ordered so that reverse edges can be detected as duplicates
    if (src > dst) {
      std::swap (src, dst);
    }

    assert (src <= dst);
  }

  // equals function purely based on src and dst to detect duplicates
  friend bool operator == (const InEdge& left, const InEdge& right) {
    return (left.src == right.src) && (left.dst == right.dst);
  }

  struct Hash {

    // hash function purely based on src and dst to find and remove duplicates
    size_t operator () (const InEdge& edge) const {
      constexpr unsigned shift = (sizeof(size_t) * 8) >> 1;
      return (static_cast<size_t>(edge.src) << shift) ^ static_cast<size_t>(edge.dst);
    }
  };

};

struct Edge: public InEdge {
  unsigned id;

  Edge (unsigned _id, int _src, int _dst, Weight_ty _weight)
    : InEdge (_src, _dst, _weight), id (_id) {}


  friend bool operator == (const Edge& left, const Edge& right) {
    return (left.id == right.id) 
      && (left.src == right.src) 
      && (left.dst == right.dst) 
      && (left.weight == right.weight);
  }

  std::string str () const {
    char s[256];
    sprintf (s, "(id=%d,src=%d,dst=%d,weight=%d)", id, src, dst, weight);
    return std::string (s);
  }

  friend std::ostream& operator << (std::ostream& out, const Edge& edge) {
    return (out << edge.str ());
  }

  struct Comparator {
    static inline int compare (const Edge& left, const Edge& right) {
      int d = left.weight - right.weight;
      return (d != 0) ? d : (left.id - right.id);
    }

    bool operator () (const Edge& left, const Edge& right) const {
      return compare (left, right) < 0;
    }
  };
};

template <typename V>
static void unionByRank_int (int rep1, int rep2, V& repVec) {
  assert (rep1 >= 0 && size_t (rep1) < repVec.size ());
  assert (rep2 >= 0 && size_t (rep2) < repVec.size ());
  assert (repVec[rep1] < 0);
  assert (repVec[rep2] < 0);

  if (repVec[rep2] < repVec[rep1]) { 
    std::swap (rep1, rep2);
  }
  assert (repVec[rep1] <= repVec[rep2]);

  repVec[rep1] += repVec[rep2];
  repVec[rep2] = rep1;
  assert (repVec[rep1] < 0);
}

template <typename V>
static void linkUp_int (int other, int master, V& repVec) {
  assert (other >= 0 && size_t (other) < repVec.size ());
  assert (master >= 0 && size_t (master) < repVec.size ());
  assert (repVec[other] < 0);
  // assert (repVec[master] < 0); // can't check this in parallel

  repVec[other] = master;
}

template <typename V>
int findPCiter_int (const int node, V& repVec) {
  assert (node >= 0 && size_t (node) < repVec.size ());

  if (repVec[node] < 0) { return node; }

  assert (repVec[node] >= 0);

  int rep = repVec[node];

  if (repVec[rep] < 0) { return rep; }

  while (repVec[rep] >= 0) { 
    rep = repVec[rep]; 
  }

  // // path compress
  // for (int n = node; n != rep;) {
    // int next = repVec[n];
    // repVec[n] = rep;
    // n = next;
  // }
  for (int n = node; n != rep; ) {
    repVec[n] = rep;
    n = repVec[n];
  }

  assert (rep >= 0 && size_t (rep) < repVec.size ());
  return rep;
}

template <typename V>
int getRep_int (const int node, const V& repVec) {
  assert (node >= 0 && size_t (node) < repVec.size ());

  if (repVec[node] < 0) { return node; }

  int rep = repVec[node];
  while (repVec[rep] >= 0) {
    rep = repVec[rep];
  }
  assert (repVec[rep] < 0);
  return rep;
}


class Kruskal {

public:

  typedef std::vector<Edge> VecEdge;

  typedef std::unordered_set<InEdge, InEdge::Hash> SetInEdge;

  static const unsigned DEFAULT_CHUNK_SIZE = 16;

protected:

  virtual const std::string getVersion () const = 0;

  //! doesn't do anything by default. Sub-classes may choose to override 
  //! in order to to specific initialization
  virtual void initRemaining (const size_t numNodes, const VecEdge& edges) { };

  virtual void runMST (const size_t numNodes, VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) = 0;


  void readGraph (const std::string& filename, size_t& numNodes, SetInEdge& edgeSet) {

    typedef galois::Graph::LC_CSR_Graph<unsigned, uint32_t> InGraph;
    typedef InGraph::GraphNode InGNode;

    InGraph ingraph;
    galois::Graph::readGraph (ingraph, filename);

    // numbering nodes 0..N-1, where N is number of nodes
    // in the graph
    unsigned idCntr = 0;
    for (InGraph::iterator n = ingraph.begin (), endn = ingraph.end ();
        n != endn; ++n) {
      ingraph.getData (*n, galois::MethodFlag::UNPROTECTED) = idCntr++;
    }
    numNodes = ingraph.size ();


    size_t numEdges = 0;
    edgeSet.clear ();

    for (InGraph::iterator n = ingraph.begin (), endn = ingraph.end ();
        n != endn; ++n) {

      unsigned src = ingraph.getData (*n, galois::MethodFlag::UNPROTECTED);


      for (InGraph::edge_iterator e = ingraph.edge_begin (src, galois::MethodFlag::UNPROTECTED),
          ende = ingraph.edge_end (src, galois::MethodFlag::UNPROTECTED); e != ende; ++e) {

        unsigned dst = ingraph.getData (ingraph.getEdgeDst (e), galois::MethodFlag::UNPROTECTED);

        if (src != dst) {
          const Weight_ty& w = ingraph.getEdgeData (e);
          InEdge ke (src, dst, w);

          std::pair<SetInEdge::iterator, bool> res = edgeSet.insert (ke);

          if (res.second) {
            ++numEdges;
          } else if (w < res.first->weight) {
            edgeSet.insert (edgeSet.erase (res.first), ke);
          }
        } else {
	  galois::Substrate::gDebug("Warning: Ignoring self edge (",
				      src, ",", dst, ",", ingraph.getEdgeData (*e), ")");
        }
      }
    }

    std::cout << "Graph read with nodes=" << ingraph.size () << ", edges=" << numEdges << std::endl;
  }
  


  virtual void readPBBSfile (const std::string& filename, size_t& numNodes, SetInEdge& edgeSet) {
    typedef unsigned NodeData;
    typedef float EdgeData;

    static const unsigned WEIGHT_SCALE = 1000000;

    std::cout << "Reading input from: " << filename << std::endl;

    // std::ifstream inFile (filename.c_str ());
    FILE* inFile = fopen (filename.c_str (), "r");

    // std::string header;
    char header[128];

    // inFile >> header;
    fscanf (inFile, "%s", header);

    // inFile.seekg (0, std::ios::beg);

    size_t numEdges = 0;
    numNodes = 0;
    edgeSet.clear ();

    // while (!inFile.eof ()) {
    while (!feof (inFile)) {
      NodeData srcIdx;
      NodeData dstIdx;
      EdgeData w;

      // inFile >> srcIdx;
      // inFile >> dstIdx;
      // inFile >> w;

      fscanf (inFile, "%d", &srcIdx);
      fscanf (inFile, "%d", &dstIdx);
      fscanf (inFile, "%g", &w);

      Weight_ty integ_wt = (WEIGHT_SCALE * w);

      if (srcIdx != dstIdx) {

        InEdge ke (srcIdx, dstIdx, integ_wt);

        std::pair<SetInEdge::iterator, bool> res = edgeSet.insert (ke);
        //edges.push_back (ke);
        if (res.second) {
          ++numEdges;
        } else if (integ_wt < res.first->weight) {
          edgeSet.insert (edgeSet.erase (res.first), ke);
        }

      } else {
          std::fprintf (stderr, "Warning: Ignoring self edge (%d, %d, %d)\n",
              srcIdx, dstIdx, integ_wt);
      }

      // find max node id;
      numNodes = std::max (numNodes, size_t (std::max (srcIdx, dstIdx)));
    }
    // inFile.close ();
    fclose (inFile);

    ++numNodes; // nodes number from 0 ... N-1


    std::cout << "PBBS graph read with nodes = " << numNodes
      << ", edges = " << numEdges << std::endl;


  }


  void writePBBSfile (const std::string& filename, const SetInEdge& edgeSet) {

    FILE* outFile = std::fopen (filename.c_str (), "w");
    assert (outFile != NULL);

    fprintf (outFile, "WeightedEdgeArray\n");

    for (SetInEdge::const_iterator i = edgeSet.begin ()
        , endi = edgeSet.end (); i != endi; ++i) {

      fprintf (outFile, "%d %d %e\n", i->src, i->dst, double (i->weight));
    }

    fclose (outFile);
  }


public:

  virtual void run (int argc, char* argv[]) {
    galois::StatManager stat;
    LonestarStart (argc, argv, name, desc, url);

    size_t numNodes;
    SetInEdge edgeSet;

    size_t mstWeight = 0;
    size_t totalIter = 0;

    galois::StatTimer t_read ("InitializeTime");

    t_read.start ();
    readGraph (filename, numNodes, edgeSet);
    // readPBBSfile (filename, numNodes, edgeSet);


    // writePBBSfile ("edgeList.pbbs", edgeSet);
    // std::exit (0);

    VecEdge edges;

    unsigned edgeIDcntr = 0;
    for (SetInEdge::const_iterator i = edgeSet.begin ()
        , endi = edgeSet.end (); i != endi; ++i) {

      edges.push_back (Edge (edgeIDcntr++, i->src, i->dst, i->weight));
    }

    assert (edges.size () == edgeSet.size ());
    t_read.stop ();


    initRemaining (numNodes, edges);


    // pre allocate memory from OS for parallel runs
    galois::preAlloc (numPages*galois::getActiveThreads ());
    galois::reportPageAlloc("MeminfoPre");
    
    galois::StatTimer t;

    t.start ();
    runMST (numNodes, edges, mstWeight, totalIter);
    t.stop ();
    galois::reportPageAlloc("MeminfoPost");

    printResults (mstWeight, totalIter);

    if (!skipVerify) {
      verify (numNodes, edgeSet, mstWeight);
    }

  }

private:

  void printResults (const size_t mstSum, const size_t iter) const {
    std::cout << getVersion () << ", MST sum=" << mstSum << ", iterations=" << iter << std::endl;
  }

  template <typename T>
  static void freeVecPtr (std::vector<T*>& vec) {
    for (typename std::vector<T*>::iterator i = vec.begin (), ei = vec.end ();
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
    Weight_ty weight;
    bool inMST;
    VecPNode_ty adj;
    VecWeight_ty adjWts;
    

    PrimNode (unsigned id): 
      id (id), 
      weight (std::numeric_limits<Weight_ty>::max ()), 
      inMST (false) {}

    void addEdge (PrimNode* pn, Weight_ty w) {
      assert (pn != NULL);
      assert (std::find (adj.begin (), adj.end (), pn) == adj.end ());

      adj.push_back (pn);
      adjWts.push_back (w);
      assert (adj.size () == adjWts.size ());
    }

    Adj_iterator_ty adj_begin () const { 
      return Adj_iterator_ty (0);
    }

    Adj_iterator_ty adj_end () const {
      return Adj_iterator_ty (adj.size ());
    }

    PrimNode* getDst (Adj_iterator_ty i) const {
      return adj[*i];
    }

    Weight_ty getWeight (Adj_iterator_ty i) const {
      return adjWts[*i];
    }

  };

  struct PrimUpdate {
    PrimNode* dst;
    Weight_ty weight;

    PrimUpdate (PrimNode* dst, Weight_ty weight)
      : dst (dst), weight (weight) {
        assert (dst != NULL);
      }


    bool operator < (const PrimUpdate& that) const {
      if (this->weight == that.weight) {
        return (this->dst->id < that.dst->id);

      } else {
        return (this->weight < that.weight);
      }
    }
  };

  size_t runPrim (const size_t numNodes, const SetInEdge& edgeSet) const {

    std::vector<PrimNode*> primNodes (numNodes, NULL);

    for (size_t i = 0; i < numNodes; ++i) {
      primNodes[i] = new PrimNode (i);
    }

    for (SetInEdge::const_iterator e = edgeSet.begin (), ende = edgeSet.end ();
        e != ende; ++e) {


      assert (primNodes[e->src] != NULL);
      assert (primNodes[e->dst] != NULL);

      // add undirected edge
      primNodes[e->src]->addEdge (primNodes[e->dst], e->weight);
      primNodes[e->dst]->addEdge (primNodes[e->src], e->weight);

    }

    std::set<PrimUpdate> workset;

    PrimNode* root = primNodes[0];
    PrimUpdate upd (root, 0);
    workset.insert (upd);


    size_t iter = 0;
    size_t mstSum = 0;

    while (!workset.empty ()) {
      ++iter;
      PrimUpdate upd = *(workset.begin ());
      workset.erase (workset.begin ());

      PrimNode& src = *(upd.dst);

      if (!src.inMST) {
        src.inMST = true;
        src.weight = upd.weight;

        mstSum += upd.weight;

        for (PrimNode::Adj_iterator_ty i = src.adj_begin (), endi = src.adj_end (); i != endi; ++i) {

          PrimNode& dst = *(src.getDst (i));
          Weight_ty wt = src.getWeight (i);

          if (!dst.inMST) {
            PrimUpdate addUpd (&dst, wt);
            workset.insert (addUpd);
          }

        }
      } // end if;
    }

    std::cout << "Number of iterations taken by Prim = " << iter << std::endl;

    freeVecPtr (primNodes);
    return mstSum;
  }

  bool verify (const size_t numNodes, const SetInEdge& edgeSet, const size_t kruskalSum) const {
    galois::StatTimer pt("PrimTime");
    pt.start ();
    size_t primSum = runPrim (numNodes, edgeSet);
    pt.stop ();

    if (primSum != kruskalSum) {
      std::cerr << "ERROR. Incorrect MST weight=" << kruskalSum 
        << ", weight computed by Prim is=" << primSum << std::endl;
      abort ();

    } else {
      std::cout << "OK. Correct MST weight=" << kruskalSum << std::endl;
    }

    return false;
  }
};


} // namespace kruskal


#endif // _KRUSKAL_H_ 
