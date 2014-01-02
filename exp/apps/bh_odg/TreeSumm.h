#ifndef BH_TREE_SUM_H
#define BH_TREE_SUM_H

#include "Point.h"
#include "Octree.h"

#include "Galois/config.h"
#include "Galois/Runtime/ROBexecutor.h"
#include "Galois/Runtime/LevelExecutor.h"
#include "Galois/Runtime/KDGtwoPhase.h"

#include <boost/iterator/transform_iterator.hpp>

#ifdef HAVE_CILK
#include <cilk/reducer_opadd.h>
#endif

namespace bh {

template <typename B>
struct TypeDefHelper {
  using Base_ty = B;
  using TreeNode = Octree<B>;
  using InterNode = OctreeInternal<B>;
  using Leaf = Body<B>;
  using VecTreeNode = std::vector<TreeNode*>;
  using VecInterNode = std::vector<InterNode*>;
};


struct TreeSummarizeSerial: public TypeDefHelper<SerialNodeBase> {

  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {
    root->mass = recurse(root);
  }

private:
  double recurse(InterNode* node) const {
    double mass = 0.0;
    Point accum;

    node->compactChildren ();
    
    for (int i = 0; i < 8; i++) {
      TreeNode* child = node->getChild (i);
      if (child == NULL)
        break;

      double m;
      const Point* p;
      if (child->isLeaf()) {
        Leaf* n = static_cast<Leaf*>(child);
        m = n->mass;
        p = &n->pos;
      } else {
        InterNode* n = static_cast<InterNode*>(child);
        m = recurse(n);
        p = &n->pos;
      }

      mass += m;
      for (int j = 0; j < 3; j++) 
        accum[j] += (*p)[j] * m;
    }

    node->mass = mass;
    
    if (mass > 0.0) {
      double inv_mass = 1.0 / mass;
      for (int j = 0; j < 3; j++)
        node->pos[j] = accum[j] * inv_mass;
    }

    return mass;
  }
};

#ifdef HAVE_CILK
struct TreeSummarizeCilk: public TypeDefHelper<SerialNodeBase> {
  TreeSummarizeCilk() {
    if (!Galois::Runtime::LL::EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD")) {
      GALOIS_DIE("set environment variable GALOIS_DO_NOT_BIND_MAIN_THREAD");
    }
  }

  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {
    root->mass = recurse(root);
  }

private:
  double recurse(InterNode* node) const {
    cilk::reducer_opadd<double> mass;
    cilk::reducer_opadd<Point> accum;

    node->compactChildren ();
    
    cilk_for (int i = 0; i < 8; i++) {
      TreeNode* child = node->getChild (i);
      if (child == NULL)
        continue;

      double m;
      const Point* p;
      if (child->isLeaf()) {
        Leaf* n = static_cast<Leaf*>(child);
        m = n->mass;
        p = &n->pos;
      } else {
        InterNode* n = static_cast<InterNode*>(child);
        m = recurse(n);
        p = &n->pos;
      }

      mass += m;
      accum += (*p * m);
    }

    node->mass = mass.get_value();

    if (node->mass > 0.0) {
      Point a = accum.get_value();
      a *= 1.0 / node->mass;
      node->pos = a;
    }

    return node->mass;
  }
};
#else
struct TreeSummarizeCilk: public TypeDefHelper<SerialNodeBase> {
  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {
    GALOIS_DIE("not implemented due to missing cilk support");
  }
};

#endif

template<typename B>
GALOIS_ATTRIBUTE_PROF_NOINLINE static void treeCompute (OctreeInternal<B>* node) {

  assert ((node != NULL) && (!node->isLeaf ()));

  double massSum = 0.0;
  Point accum;

  node->compactChildren ();

  for (unsigned i = 0; i < 8; ++i) {
    Octree<B>* child = node->getChild (i);

    if (child == NULL) {
      break;
    }

    massSum += child->mass;

    for (unsigned j = 0; j < 3; ++j) {
      accum [j] += child->mass * child->pos[j];
    }

  } // end for child

  node->mass = massSum;

  if (massSum > 0.0) {
    double invSum = 1.0 / massSum;

    for (unsigned j = 0; j < 3; ++j) {
      node->pos [j] = accum [j] * invSum;
    }
  }

}

struct TreeSummarizeODG: public TypeDefHelper<SerialNodeBase> {

  typedef Galois::GAtomic<unsigned> UnsignedAtomic;
  static const unsigned CHUNK_SIZE = 64;
  typedef Galois::WorkList::dChunkedFIFO<CHUNK_SIZE, unsigned> WL_ty;

  struct ODGnode {
    UnsignedAtomic numChild;

    InterNode* node;
    unsigned idx;
    unsigned prtidx;

    ODGnode (InterNode* _node, unsigned _idx, unsigned _prtidx) 
      : numChild (0), node (_node), idx (_idx), prtidx (_prtidx)

    {}

  };

  void fillWL (InterNode* root, std::vector<ODGnode>& odgNodes, WL_ty& wl) const {

    ODGnode root_wrap (root, 0, 0);
    odgNodes.push_back (root_wrap);

    std::deque<unsigned> fifo;
    fifo.push_back (root_wrap.idx);

    unsigned idCntr = 1; // already generated the root;

    while (!fifo.empty ()) {

      unsigned nid = fifo.front ();
      fifo.pop_front ();

      InterNode* node = odgNodes[nid].node;
      assert ((node != NULL) && (!node->isLeaf ()));

      bool allLeaves = true;
      for (unsigned i = 0; i < 8; ++i) {
        if (node->getChild (i) != NULL) {

          if (!node->getChild (i)->isLeaf ()) {
            allLeaves = false;

            InterNode* child = static_cast <InterNode*> (node->getChild (i));

            ODGnode c_wrap (child, idCntr, nid);
            ++idCntr;

            odgNodes.push_back (c_wrap);
            fifo.push_back (c_wrap.idx);

            // also count the number of children
            ++(odgNodes [nid].numChild);
          }

        }
      }

      if (allLeaves) {
        wl.push (nid);
      }

    }

  }

  struct SummarizeOp {

    std::vector <ODGnode>& odgNodes;

    SummarizeOp (std::vector <ODGnode>& _odgNodes) 
      : odgNodes (_odgNodes)
    {}

    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE static void addToWL (C& lwl, unsigned v) {
      lwl.push (v);
    }

    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void updateODG (unsigned nid, C& lwl) {

      unsigned prtidx = odgNodes[nid].prtidx;

      if (nid != 0) { // not root

        unsigned x = --(odgNodes[prtidx].numChild);

        assert (x < 8);

        if (x == 0) {
          // lwl.push (prtidx);
          addToWL (lwl, prtidx);
        }
      } else {
        assert (nid == prtidx && nid == 0);
      }
    }

    template <typename ContextTy>
    void operator () (unsigned nid, ContextTy& lwl) {
      assert (odgNodes[nid].numChild == 0);

      treeCompute (odgNodes[nid].node);

      updateODG (nid, lwl);
    }

  };



  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {
    WL_ty wl;
    std::vector<ODGnode> odgNodes;

    Galois::StatTimer t_fill_wl ("Time to fill worklist for tree summarization: ");

    t_fill_wl.start ();
    fillWL (root, odgNodes, wl);
    t_fill_wl.stop ();

    Galois::StatTimer t_feach ("Time taken by for_each in tree summarization");

    t_feach.start ();
    Galois::Runtime::beginSampling ();
    // Galois::for_each_wl<Galois::Runtime::WorkList::ParaMeter<WL_ty> > (wl, SummarizeOp (odgNodes), "tree_summ");
    Galois::for_each_wl (wl, SummarizeOp (odgNodes), "tree_summ");
    Galois::Runtime::endSampling ();
    t_feach.stop ();

  }

};


struct TreeSummarizeLevelByLevel: public TypeDefHelper<SerialNodeBase> {

  void fillWL (InterNode* root, std::vector<std::vector<InterNode*> >& levelWL) const {
    levelWL.push_back (std::vector<InterNode*> ());

    levelWL[0].push_back (root);

    unsigned currLevel = 0;

    while (!levelWL[currLevel].empty ()) {
      unsigned nextLevel = currLevel + 1;

      // creating vector for nextLevel
      levelWL.push_back (std::vector<InterNode*> ());

      for (std::vector<InterNode*>::const_iterator i = levelWL[currLevel].begin ()
          , ei = levelWL[currLevel].end (); i != ei; ++i) {

        for (unsigned c = 0; c < 8; ++c) {
          TreeNode* child = (*i)->getChild (c);
          if (child != NULL) {

            if (!child->isLeaf ()) {
              levelWL[nextLevel].push_back (static_cast<InterNode*> (child));
            }
          }
        } // for child c

      }

      ++currLevel;
    }

  }


  struct SummarizeOp {

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (InterNode* node) const {
      treeCompute (node);
    }

  };



  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {
    const bool USE_PARAMETER = false;


    std::vector<std::vector<InterNode*> > levelWL;


    Galois::StatTimer t_fill_wl ("Time to fill worklist for tree summarization: ");

    t_fill_wl.start ();
    fillWL (root, levelWL);
    t_fill_wl.stop ();


    size_t iter = 0;

    std::ofstream* statsFile = NULL;
    if (USE_PARAMETER) {
      statsFile = new std::ofstream ("parameter_barneshut.csv");
      (*statsFile) << "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE" << std::endl;
    }

    Galois::StatTimer t_feach ("Time taken by for_each in tree summarization");

    t_feach.start ();
    Galois::Runtime::beginSampling ();
    for (unsigned i = levelWL.size (); i > 0;) {
      
      --i; // size - 1

      if (!levelWL[i].empty ()) {
        Galois::Runtime::do_all_coupled (levelWL[i].begin (), levelWL[i].end (),
            SummarizeOp ());


        if (USE_PARAMETER) {
          unsigned step = (levelWL.size () - i - 2);
          (*statsFile) << "tree_summ, " << step << ", " << levelWL[i].size () 
            << ", " << levelWL[i].size () << std::endl;
        }
      }

      iter += levelWL[i].size ();


    }
    Galois::Runtime::endSampling ();
    t_feach.stop ();

    std::cout << "TreeSummarizeLevelByLevel: iterations = " << iter << std::endl;

    if (USE_PARAMETER) {
      delete statsFile;
    }

  }

};


struct TreeSummarizeSpeculative: public TypeDefHelper<SpecNodeBase> {

  struct VisitNhood {

    void acquire (TreeNode* n) {
      Galois::Runtime::acquire (n, Galois::CHECK_CONFLICT);
    }

    template <typename C>
    void operator () (InterNode* node, C& ctx) {

      assert (!node->isLeaf ());
      acquire (node);

      for (unsigned i = 0; i < 8; ++i) {
        TreeNode* c = node->getChild (i);
        if (c != NULL) {
          acquire (c);
        }
      }
    }
  };


  struct OpFunc {

    template <typename C>
    void operator () (InterNode* node, C& ctx) {

      double orig_mass = node->mass;
      Point orig_pos = node->pos;

      auto restore = [node, orig_mass, orig_pos] (void) {
        node->pos = orig_pos;
        node->mass = orig_mass;
      };

      ctx.addUndoAction (restore);

      treeCompute (node);

    }
  };

  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {

    Galois::StatTimer t_copy_to_vec ("time for copying nodes into a vector");

    VecInterNode nodes;

    t_copy_to_vec.start ();
    copyToVecInterNodes (root, nodes);
    t_copy_to_vec.stop ();

    Galois::StatTimer t_feach ("time for speculative for_each");

    t_feach.start ();
    Galois::Runtime::for_each_ordered_rob (
        nodes.begin (), nodes.end (),
        LevelComparator<TreeNode> (), VisitNhood (), OpFunc ());
    t_feach.stop ();

  }
};


struct TreeSummarizeTwoPhase: public TreeSummarizeSpeculative {

  using Base = TreeSummarizeSpeculative;

  struct OpFunc {
    
    typedef char tt_does_not_need_push;
    static const unsigned CHUNK_SIZE = 1024;

    template <typename C>
    void operator () (InterNode* node, C& ctx) {
      treeCompute (node);
    }
  };

  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {

    Galois::StatTimer t_copy_to_vec ("time for copying nodes into a vector");

    VecInterNode nodes;

    t_copy_to_vec.start ();
    copyToVecInterNodes (root, nodes);
    t_copy_to_vec.stop ();

    Galois::StatTimer t_feach ("time for two-phase for_each");

    t_feach.start ();
    Galois::Runtime::for_each_ordered_2p_win (
        nodes.begin (), nodes.end (),
        LevelComparator<TreeNode> (), Base::VisitNhood (), OpFunc ());
    t_feach.stop ();

  }
};

struct TreeSummarizeLevelExec: public TypeDefHelper<LevelNodeBase> {

  struct VisitNhood {

    template <typename C>
    void operator () (const InterNode* node, C& ctx) const {}
  };


  struct OpFunc {

    static const unsigned CHUNK_SIZE = 512;

    template <typename C>
    void operator () (InterNode* node, C& ctx) const {
      treeCompute (node);
    }
  };

  struct GetLevel {
    unsigned operator () (const InterNode* node) const {
      return node->level;
    }
  };

  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {

    Galois::StatTimer t_copy_to_vec ("time for copying nodes into a vector");

    VecInterNode nodes;

    t_copy_to_vec.start ();
    copyToVecInterNodes (root, nodes);
    t_copy_to_vec.stop ();

    Galois::StatTimer t_feach ("time for level-by-level for_each");

    t_feach.start ();
    Galois::Runtime::for_each_ordered_level (
        nodes.begin (), nodes.end (),
        GetLevel (), std::greater<unsigned> (), VisitNhood (), OpFunc ());
    t_feach.stop ();

  }
};

struct TreeSummarizeKDGsemi: public TypeDefHelper<KDGNodeBase> {

  struct OpFunc {
    InterNode* root;

    explicit OpFunc (InterNode* root): root (root) {}

    template <typename C>
    void operator () (InterNode* node, C& ctx) {
      assert (node->numChild == 0);

      // std::cout << "Processing node: " << node << std::endl;

      treeCompute (node);

      if (node != root) {
        InterNode* p = static_cast<InterNode*> (node->parent);
        assert (p != nullptr);

        assert (p->numChild > 0);

        unsigned x = --(p->numChild);
        assert (x < 8);

        if (x == 0) {
          ctx.push (p);
        }

      } else {
        assert (node->parent == nullptr);
      }
    }
  };

  static void checkTree (InterNode* root) {
    unsigned counted = 0;
    for (unsigned i = 0; i < 8; ++i) {
      TreeNode* c = root->getChild (i);

      if (c != nullptr) {
        assert (c->parent == root);
        ++counted;

        if (!c->isLeaf ()) {
          checkTree (static_cast<InterNode*> (c));
        }
      }
    }

    assert (counted == root->numChild);
  }

  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {

    static const unsigned CHUNK_SIZE = 64;
    typedef Galois::WorkList::dChunkedFIFO<CHUNK_SIZE, InterNode*> WL_ty;

    checkTree (root);

    Galois::StatTimer t_fill_wl ("Time to fill worklist for tree summarization: ");

    t_fill_wl.start ();
    WL_ty wl;
    Galois::do_all (bodbeg, bodend, 
        [&wl] (Leaf* l) {
          InterNode* p = static_cast<InterNode*> (l->parent);
          unsigned c = --(p->numChild);

          if (c == 0) {
            // std::cout << "Adding to wl: " << p << ", with numChild=" << p->numChild << std::endl;
            wl.push (p);
          }
        },
        Galois::loopname("fill_init_wl"));
    t_fill_wl.stop ();

    Galois::StatTimer t_feach ("Time taken by for_each in tree summarization");

    t_feach.start ();
    Galois::for_each_wl (wl, OpFunc (root), "tree_summ");
    t_feach.stop ();

  }
};



} // end namespace bh

#endif //  BH_TREE_SUM_H
