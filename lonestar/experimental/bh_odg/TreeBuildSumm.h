#ifndef BH_TREE_SUM_H
#define BH_TREE_SUM_H

#include "Point.h"
#include "Octree.h"

#include "galois/Bag.h"
#include "galois/runtime/ROBexecutor.h"
#include "galois/runtime/OrderedSpeculation.h"
#include "galois/runtime/LevelExecutor.h"
#include "galois/runtime/KDGtwoPhase.h"
#include "galois/runtime/DAGexec.h"
#include "galois/runtime/DAGexecAlt.h"
#include "galois/runtime/TreeExec.h"
#include "galois/runtime/Sampling.h"

#include <boost/iterator/transform_iterator.hpp>

#ifdef HAVE_CILK
#include <cilk/reducer_opadd.h>
#endif

namespace bh {

// TODO: use range type instead of iterator pairs
//

template <typename B>
void checkTreeBuildRecursive (unsigned& leafCount, const OctreeInternal<B>* node) {
  assert (!node->isLeaf ());

  for (unsigned i = 0; i < 8; ++i) {
    if (node->getChild (i) != nullptr) {

      if (i != getIndex (node->pos, node->getChild (i)->pos)) { 
        GALOIS_DIE ("child pos out of bounding box");
      }

      if (!node->getChild (i)->isLeaf ()) { 
        checkTreeBuildRecursive (
            leafCount, 
            static_cast<const OctreeInternal<B>*> (node->getChild (i)));

      } else {
        ++leafCount;
      }
    }
  }
}

template <typename B>
void checkTreeBuild (const OctreeInternal<B>* root, const BoundingBox& box, const unsigned numBodies) {


  // 1. num leaves == numBodies
  // 2. every node's position is within its bounding box

  unsigned leafCount = 0;
  checkTreeBuildRecursive (leafCount, root);

  if (leafCount != numBodies) {
    GALOIS_DIE ("mismatch in num-leaves and num-bodies");
  }
}

template <typename B, typename TreeAlloc>
struct DestroyTree {

  struct TopDown {

    template <typename C>
    void operator () (OctreeInternal<B>* node, C& wl) const {
      assert (node != nullptr && !node->isLeaf ());

      for (unsigned i = 0; i < 8; ++i) {
        Octree<B>* child = node->getChild (i);
        if (child != nullptr && !child->isLeaf ()) {
          wl.spawn (static_cast<OctreeInternal<B>*> (child));
        }
      }
    }
  };

  struct BottomUp {

    TreeAlloc& treeAlloc;

    void operator () (OctreeInternal<B>* node) {
      assert (node != nullptr && !node->isLeaf ());
      this->treeAlloc.destroy (node);
      this->treeAlloc.deallocate (node, 1);
    }
  };


  void operator () (OctreeInternal<B>* root, TreeAlloc& treeAlloc) {

    galois::StatTimer t_destroy ("time to destroy the tree recursively: ");

    t_destroy.start ();
    galois::runtime::for_each_ordered_tree (
        root,
        TopDown (),
        BottomUp {treeAlloc},
        "octree-destroy");


    t_destroy.stop ();
  }


};

template <typename B, typename TreeAlloc>
void destroyTree (OctreeInternal<B>* root, TreeAlloc& treeAlloc) {
  DestroyTree<B, TreeAlloc> d;
  d (root, treeAlloc);
}


template <typename B>
struct BuildTreeSerial {

  template <typename TreeAlloc, typename InternalNodes>
  struct BuildOperator {
    // NB: only correct when run sequentially
    typedef int tt_does_not_need_stats;

    TreeAlloc& treeAlloc;
    InternalNodes& internalNodes;
    OctreeInternal<B>* root;
    double root_radius;

    BuildOperator (
        TreeAlloc& treeAlloc, 
        InternalNodes& internalNodes, 
        OctreeInternal<B>* _root, 
        double radius) 
      :

        treeAlloc (treeAlloc),
        internalNodes (internalNodes),
        root(_root),
        root_radius(radius)

      {}


    void operator () (Body<B>* b) {
      insert(b, root, root_radius, treeAlloc, internalNodes);
    }

    template<typename Context>
    void operator()(Body<B>* b, Context&) {
      (*this) (b);
    }

    static void insert(Body<B>* b, OctreeInternal<B>* node, double radius, TreeAlloc& treeAlloc, InternalNodes& internalNodes) {

      int index = getIndex(node->pos, b->pos);

      assert(!node->isLeaf());

      Octree<B>* child = node->getChild (index);

      if (child == NULL) {
        node->setChild (index, b);
        return;
      }

      radius *= 0.5;
      if (child->isLeaf()) {
        // Expand leaf
        Body<B>* n = static_cast<Body<B>*>(child);
        Point new_pos(node->pos);
        updateCenter(new_pos, index, radius);

        OctreeInternal<B>* new_node = treeAlloc.allocate (1);
        treeAlloc.construct (new_node, new_pos);
        internalNodes.push_back (new_node);

        assert(n->pos != b->pos);

        node->setChild (index, new_node);
        insert(b, new_node, radius, treeAlloc, internalNodes);
        insert(n, new_node, radius, treeAlloc, internalNodes);
      } else {
        OctreeInternal<B>* n = static_cast<OctreeInternal<B>*>(child);
        insert(b, n, radius, treeAlloc, internalNodes);
      }
    }


  };

  template <typename I, typename TreeAlloc, typename InternalNodes>
  OctreeInternal<B>* operator () (const BoundingBox& box, I beg, I end
      , TreeAlloc& treeAlloc, InternalNodes& internalNodes) const {

    OctreeInternal<B>* root = treeAlloc.allocate (1);
    treeAlloc.construct (root, box.center ());
    internalNodes.push_back (root);
      
    std::for_each (beg, end, 
        BuildOperator<TreeAlloc, InternalNodes> (treeAlloc, internalNodes, root, box.radius ()));

    return root;
  }

};

template <typename B>
struct BuildTreeLockFree {

  template <typename TreeAlloc, typename InternalNodes>
  struct BuildOperator {

    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;
    
    TreeAlloc& treeAlloc;
    InternalNodes& internalNodes;
    OctreeInternal<B>* root;
    double root_radius;

    void operator () (Body<B>* b) const {
      insert(b, root, root_radius, treeAlloc, internalNodes);
    }

    template<typename Context>
    void operator()(Body<B>* b, Context&) const {
      operator() (b);
    }

    static void insert (Body<B>* b, OctreeInternal<B>* node, double radius
        , TreeAlloc& treeAlloc, InternalNodes& internalNodes) {

      unsigned index = getIndex (node->pos, b->pos);
      assert (!node->isLeaf ());

      if (node->getChild (index) == nullptr) { 
        if (node->casChild (index, nullptr, b)) {
          return;
        }
      }

      radius *= 0.5;
      Octree<B>* child = node->getChild (index);
      assert (child != nullptr);

      if (child->isLeaf ()) {
        
        Point new_pos(node->pos);
        updateCenter(new_pos, index, radius);
        OctreeInternal<B>* new_node = treeAlloc.allocate (1);
        treeAlloc.construct (new_node, new_pos);

        assert (child->pos != b->pos);
        
        if (node->casChild (index, child, new_node)) {
          internalNodes.push_back (new_node);
          // successful thread inserts the replaced leaf
          insert (static_cast<Body<B>*> (child), new_node, radius, treeAlloc, internalNodes);

        } else {
          treeAlloc.destroy (new_node);
          treeAlloc.deallocate (new_node, 1);
          new_node = nullptr;
        }

        child = node->getChild (index);
        assert (child != nullptr && !child->isLeaf ());

        insert (b, static_cast<OctreeInternal<B>*> (child), radius, treeAlloc, internalNodes);

      } else {
        OctreeInternal<B>* m = static_cast<OctreeInternal<B>*>(child);
        insert(b, m, radius, treeAlloc, internalNodes);
      }


    }


  };

  template <typename I, typename TreeAlloc, typename InternalNodes>
  OctreeInternal<B>* operator () (const BoundingBox& box, I beg, I end
      , TreeAlloc& treeAlloc, InternalNodes& internalNodes) const {

    OctreeInternal<B>* root = treeAlloc.allocate (1);
    treeAlloc.construct (root, box.center ());
    internalNodes.push_back (root);

    typedef galois::worklists::dChunkedFIFO<64> WL_ty;
      
    galois::do_all (beg, end, 
        BuildOperator<TreeAlloc, InternalNodes> {treeAlloc, internalNodes, root, box.radius ()},
        galois::steal<true>());

    return root;
  }
};


template <typename B>
struct TypeDefHelper {
  using Base_ty = B;
  using TreeNode = Octree<B>;
  using InterNode = OctreeInternal<B>;
  using Leaf = Body<B>;
};


template <typename B=SerialNodeBase>
// struct SummarizeTreeSerial: public TypeDefHelper<B> {
struct SummarizeTreeSerial {

  // typedef TypeDefHelper<B> Super;
  // typedef typename Super::TreeNode TreeNode;
  // typedef typename Super::InterNode InterNode;
  // typedef typename Super::Leaf Leaf;
  typedef B Base_ty;
  typedef Octree<B> TreeNode;
  typedef OctreeInternal<B> InterNode;
  typedef Body<B> Leaf;

  template <typename I, typename InternalNodes>
  void operator () (InterNode* root, I bodbeg, I bodend, InternalNodes& internalNodes) const {
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
    if (!galois::runtime::LL::EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD")) {
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

      Point tmp (*p);
      tmp *= m;
      accum += tmp;
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


struct TreeSummarizeODG: public TypeDefHelper<SerialNodeBase> {

  typedef galois::GAtomic<unsigned> UnsignedAtomic;
  static const unsigned CHUNK_SIZE = 64;
  typedef galois::worklists::dChunkedFIFO<CHUNK_SIZE, unsigned> WL_ty;

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

      summarizeNode (odgNodes[nid].node);

      updateODG (nid, lwl);
    }

  };



  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {
    WL_ty wl;
    typedef galois::worklists::ExternalReference<WL_ty> WL;
    typedef typename WL_ty::value_type value_type;
    value_type* it = nullptr;
    std::vector<ODGnode> odgNodes;

    galois::StatTimer t_fill_wl ("Time to fill worklist for tree summarization: ");

    t_fill_wl.start ();
    fillWL (root, odgNodes, wl);
    t_fill_wl.stop ();

    galois::StatTimer t_feach ("Time taken by for_each in tree summarization");

    galois::runtime::beginSampling ();
    t_feach.start ();

    // galois::for_each_wl<galois::runtime::worklists::ParaMeter<WL_ty> > (wl, SummarizeOp (odgNodes), "tree_summ");
    galois::for_each(it, it, SummarizeOp (odgNodes), galois::loopname("tree_summ"), galois::wl<WL>(&wl));
    t_feach.stop ();
    galois::runtime::endSampling ();

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
      summarizeNode (node);
    }

  };



  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {
    const bool USE_PARAMETER = false;


    std::vector<std::vector<InterNode*> > levelWL;


    galois::StatTimer t_fill_wl ("Time to fill worklist for tree summarization: ");

    t_fill_wl.start ();
    fillWL (root, levelWL);
    t_fill_wl.stop ();


    size_t iter = 0;

    std::ofstream* statsFile = NULL;
    if (USE_PARAMETER) {
      statsFile = new std::ofstream ("parameter_barneshut.csv");
      (*statsFile) << "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE" << std::endl;
    }

    galois::StatTimer t_feach ("Time taken by for_each in tree summarization");

    galois::runtime::beginSampling ();
    t_feach.start ();
    for (unsigned i = levelWL.size (); i > 0;) {
      
      --i; // size - 1

      if (!levelWL[i].empty ()) {
        galois::runtime::do_all_coupled (
            galois::runtime::makeStandardRange (levelWL[i].begin (), levelWL[i].end ()),
            SummarizeOp (), 
            std::make_tuple (
              galois::loopname("level-hand"), 
              galois::chunk_size<32> ()));


        if (USE_PARAMETER) {
          unsigned step = (levelWL.size () - i - 2);
          (*statsFile) << "tree_summ, " << step << ", " << levelWL[i].size () 
            << ", " << levelWL[i].size () << std::endl;
        }
      }

      iter += levelWL[i].size ();


    }
    t_feach.stop ();
    galois::runtime::endSampling ();

    std::cout << "TreeSummarizeLevelByLevel: iterations = " << iter << std::endl;

    if (USE_PARAMETER) {
      delete statsFile;
    }

  }

};



struct TreeSummarizeSpeculative: public TypeDefHelper<SpecNodeBase> {

  struct VisitNhood {
    static const unsigned CHUNK_SIZE = 32;

    void acquire (TreeNode* n, galois::MethodFlag f) {
      galois::runtime::acquire (n, f);
    }

    template <typename C>
    void operator () (InterNode* node, C& ctx) {

      assert (!node->isLeaf ());
      acquire (node, galois::MethodFlag::WRITE);

      for (unsigned i = 0; i < 8; ++i) {
        TreeNode* c = node->getChild (i);
        if (c != NULL) {
          acquire (c, galois::MethodFlag::READ);
        }
      }
    }
  };


  template <bool useSpec>
  struct OpFunc {
    typedef char tt_does_not_need_push;
    static const unsigned CHUNK_SIZE = 32;


    template <typename C>
    void operator () (InterNode* node, C& ctx) const {

      if (useSpec) {
        double orig_mass = node->mass;
        Point orig_pos = node->pos;

        auto restore = [node, orig_mass, orig_pos] (void) {
          node->pos = orig_pos;
          node->mass = orig_mass;
        };

        ctx.addUndoAction (restore);
      }

      summarizeNode (node);

    }
  };

  template <typename I, typename InternalNodes>
  void operator () (InterNode* root, I bodbeg, I bodend, InternalNodes& internalNodes) const {

    // galois::runtime::for_each_ordered_spec (
        // galois::runtime::makeLocalRange (internalNodes),
        // LevelComparator<TreeNode> (), 
        // VisitNhood (), 
        // OpFunc<true> (),
        // std::make_tuple (
          // galois::loopname ("tree_summ_spec")));

    std::abort();
  }
};


struct TreeSummarizeTwoPhase: public TreeSummarizeSpeculative {

  using Base = TreeSummarizeSpeculative;

  template <typename I, typename InternalNodes>
  void operator () (InterNode* root, I bodbeg, I bodend, InternalNodes& internalNodes) const {

    galois::runtime::for_each_ordered_ikdg (
        galois::runtime::makeLocalRange (internalNodes),
        LevelComparator<TreeNode> (), 
        Base::VisitNhood (), 
        Base::OpFunc<false> (),
        std::make_tuple (
          galois::loopname ("tree_summ_ikdg")));

  }
};

struct TreeSummarizeDataDAG: public TreeSummarizeTwoPhase {
  using Base = TreeSummarizeTwoPhase;

  template <typename I, typename InternalNodes>
  void operator () (InterNode* root, I bodbeg, I bodend, InternalNodes& internalNodes) const {

    galois::runtime::for_each_ordered_dag (
        galois::runtime::makeLocalRange (internalNodes), 
        LevelComparator<TreeNode> (), Base::VisitNhood (), Base::OpFunc<false> ());
    // galois::runtime::for_each_ordered_dag_alt (
        // galois::runtime::makeLocalRange (internalNodes), 
        // LevelComparator<TreeNode> (), Base::VisitNhood (), Base::OpFunc<false> ());

  }

};


struct TreeSummarizeLevelExec: public TypeDefHelper<LevelNodeBase> {

  struct VisitNhood {

    template <typename C>
    void operator () (const InterNode* node, C& ctx) const {}
  };


  struct OpFunc {

    typedef int tt_does_not_need_push;
    typedef char tt_does_not_need_aborts;
    static const unsigned CHUNK_SIZE = 32;

    template <typename C>
    void operator () (InterNode* node, C& ctx) const {
      summarizeNode (node);
    }
  };

  struct GetLevel {
    unsigned operator () (const InterNode* node) const {
      return node->level;
    }
  };

  template <typename I, typename InternalNodes>
  void operator () (InterNode* root, I bodbeg, I bodend, InternalNodes& internalNodes) const {
    galois::StatTimer tt;
    tt.start();
    galois::runtime::for_each_ordered_level (
        galois::runtime::makeLocalRange (internalNodes),
        GetLevel (), std::greater<unsigned> (), VisitNhood (), OpFunc ());
    tt.stop();

  }
};

struct TreeSummarizeKDGhand: public TypeDefHelper<KDGNodeBase> {

  // TODO: add flags here for no-conflicts
  struct OpFunc {
    typedef int tt_does_not_need_aborts;

    template <typename C>
    void operator () (InterNode* node, C& ctx) {

      if (node->numChild != 0) {
        // continue visit top-down
        for (unsigned i = 0; i < 8; ++i) {
          TreeNode* c = node->getChild (i);
          if (c != nullptr && !c->isLeaf ()) {
            ctx.push (static_cast<InterNode*> (c));
          }
        }

      } else { 
        // start going up
        summarizeNode (node);

        InterNode* p = static_cast<InterNode*> (node->parent);
        if (p != nullptr) {
          unsigned x = --(p->numChild);
          if (x == 0) {
            ctx.push (p);
          }
        }
      }
      // std::cout << "Processing node: " << node << std::endl;
    }
  };

  static void checkTree (InterNode* node) {
    unsigned counted = 0;
    for (unsigned i = 0; i < 8; ++i) {
      TreeNode* c = node->getChild (i);

      if (c != nullptr && !c->isLeaf ()) {
        assert (c->parent == node);
        ++counted;
        checkTree (static_cast<InterNode*> (c));
      }
    }

    assert (counted == node->numChild);
  }

  template <typename I, typename InternalNodes>
  void operator () (InterNode* root, I bodbeg, I bodend, InternalNodes& internalNodes) const {

    static const unsigned CHUNK_SIZE = 2;
    // typedef galois::worklists::dChunkedLIFO<CHUNK_SIZE, InterNode*> WL_ty;
    typedef galois::worklists::AltChunkedLIFO<CHUNK_SIZE, InterNode*> WL_ty;

    if (!skipVerify) {
      std::cout << "KDG hand checking the tree. Timing may be off" << std::endl;
      checkTree (root);
    }

    galois::for_each (
        root,
        OpFunc (),
        galois::loopname ("kdg-hand"),
        galois::wl<WL_ty> ());


    // galois::StatTimer t_fill_wl ("Time to fill worklist for tree summarization: ");
// 
    // t_fill_wl.start ();
    // WL_ty wl;
    // galois::do_all_local (internalNodes,
        // [&wl] (InterNode* n) {
          // unsigned c = n->numChild;
// 
          // if (c == 0) {
            // // std::cout << "Adding to wl: " << p << ", with numChild=" << p->numChild << std::endl;
            // wl.push (n);
          // }
        // },
        // galois::loopname("fill_init_wl"));
    // t_fill_wl.stop ();

    // galois::for_each_wl (wl, OpFunc (root), "tree_summ");

  }
};




template <typename B, typename PartList>
struct WorkItemSinglePass {
  typedef typename PartList::iterator Iter;
  typedef OctreeInternal<B> Node_ty;

  OctreeInternal<B>* node;
  double radius;
  PartList* partList;
  Iter beg;
  Iter end;
};

template <typename PartList>
struct PartitionSinglePass {

  typedef galois::FixedSizeAllocator<PartList> PartListAlloc;

  PartListAlloc partListAlloc;
  
  template <typename WorkItem>
  void operator () (const WorkItem& w, WorkItem* child) {

    for (unsigned i = 0; i < 8; ++i) {
      PartList* list = partListAlloc.allocate (1);
      new (list) PartList ();
      child[i].partList = list;
    }

    for (auto i = w.beg, i_end = w.end; 
        i != i_end; ++i) {
      unsigned index = getIndex (w.node->pos, (*i)->pos);
      child[index].partList->push_back (*i);
    }

    // clean up child array
    for (unsigned i = 0; i < 8; ++i) {
      child[i].beg = child[i].partList->begin ();
      child[i].end = child[i].partList->end ();

      if (child[i].partList->empty ()) {
        partListAlloc.destroy (child[i].partList);
        partListAlloc.deallocate (child[i].partList, 1);
      } 
    }

    // delete the old list
    if (w.partList != nullptr) {
      partListAlloc.destroy (w.partList);
      partListAlloc.deallocate (w.partList, 1);
      const_cast<WorkItem&> (w).partList = nullptr;
    }
  }
};

// TODO: partitionMultiPassImpl


namespace recursive {

  template <bool USING_CILK, typename WorkItem, typename TreeAlloc, typename Partitioner, typename ForkJoinHandler>
  void buildSummRecurImpl (WorkItem& w, TreeAlloc& treeAlloc, Partitioner& partitioner,
      ForkJoinHandler& fjh) {

    assert (w.beg != w.end);

    auto next = w.beg; ++next;
    assert (next != w.end);

    WorkItem child[8];
    partitioner (w, child);

    auto loop_body = [&] (unsigned i) {
      if( child[i].beg != child[i].end) {
        auto next = child[i].beg; ++next;

        if (next == child[i].end) { // size 1
          w.node->setChild (i, *(child[i].beg));

        } else {
          Point new_pos = w.node->pos;
          double radius = w.radius / 2;
          updateCenter (new_pos, i, radius);
          typename WorkItem::Node_ty* internal = treeAlloc.allocate (1);
          treeAlloc.construct (internal, new_pos);
          w.node->setChild (i, internal);

          child[i].node = internal;
          child[i].radius = radius;

          fjh.fork (child[i]);
        }

      }
    };

    if (USING_CILK) {
      cilk_for (unsigned i = 0; i < 8; ++i) {
        loop_body (i);
      }
    } else {
      for (unsigned i = 0; i < 8; ++i) {
        loop_body (i);
      }
    }

    fjh.join (w);

  }

  template <bool USING_CILK, typename B, typename ForkJoinHandler>
  void summarizeRecursive (OctreeInternal<B>* node, ForkJoinHandler& fjh) {
    assert (node != nullptr && !node->isLeaf ());

    auto loop_body = [node, &fjh] (unsigned i) {
      Octree<B>* child = node->getChild (i);
      if (child != nullptr && !child->isLeaf ()) {
        fjh.fork (static_cast<OctreeInternal<B>*> (child));
      }
    };

    if (USING_CILK) {
      cilk_for (unsigned i = 0; i < 8; ++i) {
        loop_body (i);
      }
    } else {
      for (unsigned i = 0; i < 8; ++i) {
        loop_body (i);
      }
    }

    fjh.join (node);
  }



  enum ExecType {
    USE_SERIAL,
    USE_CILK,
    USE_GALOIS
  };

  template<bool USING_CILK, typename WorkItem, typename TreeAlloc, typename Partitioner> 
  struct SerialForkJoin {

    TreeAlloc& treeAlloc;
    Partitioner& partitioner;

    void fork (WorkItem& w) {
      // printf ("calling fork\n");
      buildSummRecurImpl<USING_CILK> (w, treeAlloc, partitioner, *this);
    }

    void join (WorkItem& w) {
      // printf ("calling join");
      summarizeNode (w.node);
    }
  };

  template <bool USING_CILK, typename B>
  struct SummarizeForkJoin {
    void fork (OctreeInternal<B>* node) {
      summarizeRecursive<USING_CILK> (node, *this);
    }

    void join (OctreeInternal<B>* node) {
      summarizeNode (node);
    }
  };

  template <typename C, typename WorkItem>
  struct GaloisForkHandler {

    C& ctx;

    void fork (const WorkItem& w) {
      ctx.spawn (w);
    }

    void join (const WorkItem&) {} // handled separately in Galois
  };

  template <typename WorkItem, typename TreeAlloc, typename Partitioner>
  struct GaloisBuild {

    TreeAlloc& treeAlloc;
    Partitioner& partitioner;

    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (WorkItem& w, C& ctx) {
      GaloisForkHandler<C,WorkItem> gfh {ctx};
      buildSummRecurImpl<false> (w, this->treeAlloc, this->partitioner, gfh);

    } // end method
  };

  template <typename B>
  struct GaloisPassTopDown {

    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (OctreeInternal<B>* node, C& ctx) {
      GaloisForkHandler<C,OctreeInternal<B>* > gfh {ctx};
      summarizeRecursive<false> (node, gfh);

    } // end method
  };

  template <typename WorkItem>
  struct GaloisSummarize {

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (WorkItem& w) {
      summarizeNode (w.node);
    }
  };

  template <ExecType EXEC_TYPE>
  struct ChooseExecutor {
    template <typename WorkItem, typename TreeAlloc, typename Partitioner>
    void operator () (WorkItem& initial, TreeAlloc& treeAlloc, Partitioner& partitioner) {
      GALOIS_DIE ("not implemented");
    }
  };

  template <> struct ChooseExecutor<USE_SERIAL> {
    template <typename WorkItem, typename TreeAlloc, typename Partitioner>
    void operator () (WorkItem& initial, TreeAlloc& treeAlloc, Partitioner& partitioner) {
      SerialForkJoin<false, WorkItem, TreeAlloc, Partitioner> f {treeAlloc, partitioner};
      f.fork (initial);
    }

    template <typename B>
    void operator () (OctreeInternal<B>* root) {
      SummarizeForkJoin<true, B> f;
      f.fork (root);
    }
  };

  template <> struct ChooseExecutor<USE_CILK> {
    template <typename WorkItem, typename TreeAlloc, typename Partitioner>
    void operator () (WorkItem& initial, TreeAlloc& treeAlloc, Partitioner& partitioner) {
      SerialForkJoin<true, WorkItem, TreeAlloc, Partitioner> f {treeAlloc, partitioner};
      f.fork (initial);
    }

    template <typename B>
    void operator () (OctreeInternal<B>* root) {
      SummarizeForkJoin<true, B> f;
      f.fork (root);
    }
  };

  template <> struct ChooseExecutor<USE_GALOIS> {
    template <typename WorkItem, typename TreeAlloc, typename Partitioner>
    void operator () (WorkItem& initial, TreeAlloc& treeAlloc, Partitioner& partitioner) {

      galois::runtime::for_each_ordered_tree (
          initial,
          GaloisBuild<WorkItem, TreeAlloc, Partitioner> {treeAlloc, partitioner},
          GaloisSummarize<WorkItem> (),
          "octree-build-summarize-recursive");


    }

    template <typename B>
    void operator () (OctreeInternal<B>* root) {
      galois::runtime::for_each_ordered_tree (
          root,
          GaloisPassTopDown<B> (),
          &summarizeNode<B>,
          "octree-summarize-recursive");

    }
  };

} // end namespace recursive 


template <recursive::ExecType EXEC_TYPE>
struct BuildSummarizeRecursive: public TypeDefHelper<SerialNodeBase> {

  typedef Base_ty B;


  template <typename I, typename TreeAlloc>
  OctreeInternal<B>* operator () (const BoundingBox& box, I bodbeg, I bodend, TreeAlloc& treeAlloc) const {

    typedef galois::gdeque<Body<B>*> PartList;
    typedef WorkItemSinglePass<B, PartList> WorkItem;


    OctreeInternal<B>* root = treeAlloc.allocate (1);
    treeAlloc.construct (root, box.center ());

    PartitionSinglePass<PartList> partitioner;

    WorkItem initial = {
      root,
      box.radius (),
      nullptr,
      bodbeg,
      bodend
    };

    recursive::ChooseExecutor<EXEC_TYPE> f;
    f (initial, treeAlloc, partitioner);

    return root;
  }

};

template <recursive::ExecType EXEC_TYPE> 
struct BuildLockFreeSummarizeRecursive {

  typedef SerialNodeBase B;
  typedef B Base_ty;

  template <typename I, typename TreeAlloc>
  OctreeInternal<B>* operator () (const BoundingBox& box, I bodbeg, I bodend, TreeAlloc& treeAlloc) const {

    struct DummyBag {
      void push_back (OctreeInternal<B>* node) {}
    };
    
    DummyBag _bag;

    galois::StatTimer t_build ("Time taken by tree build: ");
    t_build.start ();
    BuildTreeLockFree<B> builder;
    OctreeInternal<B>* root = builder (box, bodbeg, bodend, treeAlloc, _bag);
    t_build.stop ();

    galois::StatTimer t_summ ("Time taken by tree summarization: ");

    galois::runtime::beginSampling();
    t_summ.start ();
    recursive::ChooseExecutor<EXEC_TYPE> f;
    f (root);
    t_summ.stop ();
    galois::runtime::endSampling();

    return root;
  }

};


template <template <typename> class BM, typename SM>
struct BuildSummarizeSeparate {

  typedef typename SM::Base_ty Base_ty;
  typedef BM<Base_ty> BuildMethod;
  typedef OctreeInternal<Base_ty> Node_ty;
  typedef galois::InsertBag<Node_ty*> InternalNodes;

  BuildMethod buildMethod;
  SM summarizeMethod;

  template <typename I, typename TreeAlloc>
  Node_ty* operator () (const BoundingBox& box, I bodbeg, I bodend, TreeAlloc& treeAlloc) const {

    InternalNodes internalNodes;

    galois::StatTimer t_build ("Time taken by tree build: ");
    t_build.start ();
    Node_ty* root = buildMethod (box, bodbeg, bodend, treeAlloc, internalNodes);
    t_build.stop ();

    if (!skipVerify) {
      printf ("WARNING: Running Tree verification routine, Timing will be off\n");
      checkTreeBuild (root, box, std::distance (bodbeg, bodend));
    }

    galois::StatTimer t_summ ("Time taken by tree summarization: ");

    t_summ.start ();
    summarizeMethod (root, bodbeg, bodend, internalNodes);
    t_summ.stop ();

    return root;
  }

};








} // end namespace bh

#endif //  BH_TREE_SUM_H
