#ifndef BH_TREE_SUM_H
#define BH_TREE_SUM_H

#include "Point.h"
#include "Octree.h"

#include "Galois/config.h"
#include "Galois/Runtime/ROBexecutor.h"
// #include "Galois/Runtime/LevelExecutor.h"
#include "Galois/Runtime/KDGtwoPhase.h"
#include "Galois/Runtime/DAG.h"

#include <boost/iterator/transform_iterator.hpp>

#ifdef HAVE_CILK
#include <cilk/reducer_opadd.h>
#else
#define cilk_for for
#endif

namespace bh {


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

template <typename B>
struct TypeDefHelper {
  using Base_ty = B;
  using TreeNode = Octree<B>;
  using InterNode = OctreeInternal<B>;
  using Leaf = Body<B>;
  using VecTreeNode = std::vector<TreeNode*>;
  using VecInterNode = std::vector<InterNode*>;
};


struct SummarizeTreeSerial: public TypeDefHelper<SerialNodeBase> {

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

      summarizeNode (odgNodes[nid].node);

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
      summarizeNode (node);
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


  template <bool useSpec>
  struct OpFunc {
    typedef char tt_does_not_need_push;
    static const unsigned CHUNK_SIZE = 1024;


    template <typename C>
    void operator () (InterNode* node, C& ctx) {

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
        LevelComparator<TreeNode> (), VisitNhood (), OpFunc<true> ());
    t_feach.stop ();

  }
};


struct TreeSummarizeTwoPhase: public TreeSummarizeSpeculative {

  using Base = TreeSummarizeSpeculative;

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
        LevelComparator<TreeNode> (), Base::VisitNhood (), Base::OpFunc<false> ());
    t_feach.stop ();

  }
};

struct TreeSummarizeDAG: public TreeSummarizeTwoPhase {
  using Base = TreeSummarizeTwoPhase;

  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {

    Galois::StatTimer t_copy_to_vec ("time for copying nodes into a vector");

    VecInterNode nodes;

    t_copy_to_vec.start ();
    copyToVecInterNodes (root, nodes);
    t_copy_to_vec.stop ();

    Galois::StatTimer t_feach ("time for data dependend DAG based for_each");

    t_feach.start ();
    Galois::Runtime::for_each_ordered_dag (
        Galois::Runtime::makeStandardRange (nodes.begin (), nodes.end ()), 
        LevelComparator<TreeNode> (), Base::VisitNhood (), Base::OpFunc<false> ());
    t_feach.stop ();

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
    static const unsigned CHUNK_SIZE = 512;

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

  template <typename I>
  void operator () (InterNode* root, I bodbeg, I bodend) const {

    Galois::StatTimer t_copy_to_vec ("time for copying nodes into a vector");

    VecInterNode nodes;

    t_copy_to_vec.start ();
    copyToVecInterNodes (root, nodes);
    t_copy_to_vec.stop ();

    Galois::StatTimer t_feach ("time for level-by-level for_each");

    t_feach.start ();
    // Galois::Runtime::for_each_ordered_level (
        // Galois::Runtime::makeStandardRange (nodes.begin (), nodes.end ()),
        // GetLevel (), std::greater<unsigned> (), VisitNhood (), OpFunc ());
    t_feach.stop ();

  }
};

struct TreeSummarizeKDGsemi: public TypeDefHelper<KDGNodeBase> {

  // TODO: add flags here for no-conflicts
  struct OpFunc {
    InterNode* root;

    explicit OpFunc (InterNode* root): root (root) {}

    template <typename C>
    void operator () (InterNode* node, C& ctx) {
      assert (node->numChild == 0);

      // std::cout << "Processing node: " << node << std::endl;

      summarizeNode (node);

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

  typedef Galois::GFixedAllocator<PartList> PartListAlloc;

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

  template <typename C, typename WorkItem>
  struct GaloisForkHandler {

    C& ctx;

    void fork (WorkItem& w) {
      ctx.push (w);
    }

    void join (WorkItem&) {} // handled separately in Galois
  };

  template <typename WorkItem, typename TreeAlloc, typename Partitioner>
  struct GaloisBuild {

    TreeAlloc& treeAlloc;
    Partitioner& partitioner;

    template <typename C>
    void operator () (WorkItem& w, C& ctx) {
      GaloisForkHandler<C,WorkItem> gfh {ctx};
      buildSummRecurImpl<false> (w, this->treeAlloc, this->partitioner, gfh);

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
  };

  template <> struct ChooseExecutor<USE_CILK> {
    template <typename WorkItem, typename TreeAlloc, typename Partitioner>
    void operator () (WorkItem& initial, TreeAlloc& treeAlloc, Partitioner& partitioner) {
      SerialForkJoin<true, WorkItem, TreeAlloc, Partitioner> f {treeAlloc, partitioner};
      f.fork (initial);
    }
  };

  template <> struct ChooseExecutor<USE_GALOIS> {
    template <typename WorkItem, typename TreeAlloc, typename Partitioner>
    void operator () (WorkItem& initial, TreeAlloc& treeAlloc, Partitioner& partitioner) {

      Galois::Runtime::for_each_ordered_tree_1p (
          initial,
          GaloisBuild<WorkItem, TreeAlloc, Partitioner> {treeAlloc, partitioner},
          GaloisSummarize<WorkItem> (),
          "octree-build-summarize-1p");


    }
  };

} // end namespace recursive 


template <recursive::ExecType EXEC_TYPE>
struct BuildSummarizeRecursive: public TypeDefHelper<SerialNodeBase> {

  typedef Base_ty B;


  template <typename TreeAlloc, typename I>
  OctreeInternal<B>* operator () (const BoundingBox& box, TreeAlloc& treeAlloc, I bodbeg, I bodend) const {

    typedef Galois::gdeque<Body<B>*> PartList;
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


template <typename BM, typename SM>
struct BuildSummarizeSeparate {

  BM builMethod;
  SM summarizeMethod;
  typedef typename SM::Base_ty Base_ty;
  typedef OctreeInternal<Base_ty> Node_ty;


  template <typename TreeAlloc, typename I>
  Node_ty* operator () (const BoundingBox& box, TreeAlloc& treeAlloc, I bodbeg, I bodend) const {

    Node_ty* root = builMethod (box, treeAlloc, bodbeg, bodend);

    if (!skipVerify) {
      printf ("Running Tree verification routine\n");
      checkTreeBuild (root, box, std::distance (bodbeg, bodend));
    }

    summarizeMethod (root, bodbeg, bodend);
    return root;
  }

};






} // end namespace bh

#endif //  BH_TREE_SUM_H
