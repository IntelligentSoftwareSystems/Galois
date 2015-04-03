#ifndef GALOIS_RUNTIME_DET_PART_INPUT_DAG
#define GALOIS_RUNTIME_DET_PART_INPUT_DAG

#include "Galois/AltBag.h"

#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/DetChromatic.h"
#include "Galois/Runtime/DoAllCoupled.h"

#include <atomic>
#include <functional>


namespace Galois {
namespace Runtime {

// TODO: instead of storing partition number, store a pointer to partition meta
// data every where

struct InputDAGdataPartInOut: public InputDAGdataInOut {

  int partition = -1;
  bool isBoundary = false;

  explicit InputDAGdataPartInOut (unsigned id): InputDAGdataInOut (id) {}
};


template <typename G, typename F, typename M>
struct InputGraphPartDAGexecutor {

  using GNode = typename G::GraphNode;
  using Bag_ty = Galois::PerThreadBag<GNode>;
  using PartAdjMatrix = std::vector<std::vector<unsigned> >;

  struct BFSpartitioner {
    using ParCounter = Galois::GAccumulator<size_t>;
    using PartCounters = std::vector<ParCounter>;

    G& graph;
    M& dagManager;
    PartAdjMatrix& adjMatrix;
    const unsigned numPart;

    PartCounters  partSizes;
    const size_t partSizeLim;

    BFSpartitioner (
        G& graph,
        M& dagManager,
        PartAdjMatrix& adjMatrix,
        const unsigned numPart)
      :
        graph (graph),
        dagManager (dagManager),
        adjMatrix (adjMatrix),
        numPart (numPart),
        partSizes (numPart),
        partSizeLim ((graph.size () + numPart)/numPart)
    {}

    template <typename R, typename B>
    void pickSources (const R& range, const size_t rangeSize, B& sources) {

      ptrdiff_t jump_size = std::max (rangeSize /(numPart), size_t (1));

      auto iter = range.begin ();
      size_t dist = 0;

      for (unsigned i = 0; i < numPart && dist < rangeSize; ++i) {


        if (partSizes[i].reduceRO () < partSizeLim) {
          auto& nd = graph.getData (*iter, Galois::MethodFlag::UNPROTECTED);
          nd.partition = i;
          partSizes [i] += 1;

          sources.push_back (*iter);
          std::advance (iter, jump_size);
          dist += jump_size;
        }
      }
    }

    template <typename I>
    void bfsTraversal (I beg, I end) {

      using WL = Galois::WorkList::dChunkedFIFO<32>;
      // using WL = Galois::WorkList::AltChunkedFIFO<8>;

      Galois::for_each (beg, end,
          [this] (GNode src, Galois::UserContext<GNode>& ctxt) {

            auto& sd = graph.getData (src, Galois::MethodFlag::UNPROTECTED);
            GALOIS_ASSERT (sd.partition != -1);
            size_t psize = partSizes[sd.partition].reduceRO ();

            // bool addMore = (psize < partSizeLim); 
            bool addMore = true;

            if (addMore) {
              auto addMoreClosure = [this, &sd, &ctxt] (GNode dst) {
                auto& dd = graph.getData (dst, Galois::MethodFlag::UNPROTECTED);

                if (dd.partition == -1) {
                  dd.partition = sd.partition;
                  partSizes[dd.partition] += 1;
                  ctxt.push (dst);
                }
              };

              dagManager.applyToAdj (src, addMoreClosure, Galois::MethodFlag::UNPROTECTED);
            }
 
            
          },
          Galois::loopname ("partition_bfs"),
          Galois::wl<WL> ());
    }

    template <typename R, typename W>
    void filterUnpartitioned (const R& range, W& unpartitioned) {

      assert (unpartitioned.empty_all ());
      Galois::Runtime::do_all_coupled (range,
          [this, &unpartitioned] (GNode n) {
            auto& nd = graph.getData (n, Galois::MethodFlag::UNPROTECTED);
            if (nd.partition == -1) {
              unpartitioned.push_back (n);
            }
          },
          "find-unpartitioned");
    }

    void partition (void) {

      Galois::StatTimer ptime ("partition time");
      ptime.start ();

      Galois::PerThreadBag<GNode, 64>* currRemaining = new PerThreadBag<GNode, 64> ();
      Galois::PerThreadBag<GNode, 64>* nextRemaining = new PerThreadBag<GNode, 64> ();

      Galois::gdeque<GNode, 64> sources;

      bool first = true;
      unsigned rounds = 0;

      do {
        ++rounds;

        sources.clear ();
        if (first) {
          pickSources (makeLocalRange (graph), graph.size (),  sources);

        } else {
          pickSources (makeLocalRange (*currRemaining), currRemaining->size_all (), sources);
        }

        bfsTraversal (sources.begin (), sources.end ());

        if (first) {
          first = false;
          filterUnpartitioned (makeLocalRange (graph), *nextRemaining);

        } else {
          filterUnpartitioned (makeLocalRange (*currRemaining), *nextRemaining);
        }

        std::swap (currRemaining, nextRemaining);
        nextRemaining->clear_all_parallel ();


      } while (!currRemaining->empty_all ());

      std::printf ("BFSpartitioner partitioned in %d rounds\n", rounds);

      delete currRemaining; currRemaining = nullptr;
      delete nextRemaining; nextRemaining = nullptr;

      ptime.stop ();

    }

    unsigned countComponents (void) {

      Galois::StatTimer cctime ("countComponents time");
      cctime.start ();

      std::vector<unsigned> componentIDs (graph.size (), 0);
      
      auto nextSource = graph.begin ();

      unsigned numComp = 0;

      while (nextSource != graph.end ()) {

        ++numComp;
        // run BFS

        componentIDs[*nextSource] = numComp;
        GNode init[] = { *nextSource };
        Galois::for_each ( &init[0], &init[1], 
            [this, &componentIDs, &numComp] (GNode n, Galois::UserContext<GNode>& ctxt) {

              assert (componentIDs[n] != 0);

              auto visitAdjClosure = [this, &numComp, &componentIDs, &ctxt] (GNode dst) {
                if (componentIDs[dst] == 0) {
                  componentIDs[dst] = numComp;
                  ctxt.push (dst);
                }
              };

              dagManager.applyToAdj (n, visitAdjClosure);

            },
            Galois::loopname ("find_component_bfs"),
            Galois::wl<Galois::WorkList::dChunkedFIFO<32> > ());
        

        // find next Source
        for (auto end = graph.end (); nextSource != end; ++nextSource) {
          if (componentIDs[*nextSource] == 0) {
            break;
          }
        }
        
      } // end while

      cctime.stop ();

      return numComp;

    }


    void initCoarseAdj (void) {

      Galois::StatTimer t("time initCoarseAdj");

      t.start ();

      PartCounters partBoundarySizes (numPart);
      for (ParCounter& c: partSizes) {
        c.reset ();
      }

      do_all_coupled (makeLocalRange (graph),
          [this, &partBoundarySizes] (GNode src) {

            bool boundary = false;

            auto& sd = graph.getData (src, Galois::MethodFlag::UNPROTECTED);

            GALOIS_ASSERT (sd.partition != -1);

            partSizes[sd.partition] += 1;

            auto visitAdjClosure = [this, &sd, &boundary] (GNode dst) {
              auto& dd = graph.getData (dst, Galois::MethodFlag::UNPROTECTED);

              GALOIS_ASSERT (dd.partition != -1);
              if (dd.partition != sd.partition) {
                boundary = true;

                if (adjMatrix [sd.partition][dd.partition] != 1) {
                  adjMatrix [sd.partition][dd.partition] = 1;
                }

                if (adjMatrix [dd.partition][sd.partition] != 1) {
                  adjMatrix [dd.partition][sd.partition] = 1;
                }
              }
            };


            dagManager.applyToAdj (src, visitAdjClosure);
            sd.isBoundary = boundary;

            if (boundary) {
              partBoundarySizes[sd.partition] += 1;
            }

            // GALOIS_ASSERT (sd.isBoundary == boundary);
           
          },
          "check_partitions");

      for (size_t i = 0; i < numPart; ++i) {
        size_t total = partSizes[i].reduceRO ();
        size_t boundary = partBoundarySizes[i].reduceRO ();
        assert (total >= boundary);
        size_t inner = total - boundary;
        std::printf ("partition %zd, size =%zd, boundary=%zd, inner=%zd\n"
            , i, total, boundary, inner);
      }
      
      t.stop ();
    }
  }; // end class BFSpartitioner

  struct BlockPartitioner: public BFSpartitioner {
    using Base = BFSpartitioner;
    
    BlockPartitioner (
        G& graph,
        M& dagManager,
        PartAdjMatrix& adjMatrix,
        const unsigned numPart)
      : 
        Base (graph, dagManager, adjMatrix, numPart)
    {}

    void partition (void) {
      on_each_impl (
          [this] (const unsigned tid, const unsigned numT) {
            unsigned partPerThread = (Base::numPart + numT - 1) / numT;
            unsigned pbeg = tid * partPerThread;

            size_t size = 0;
            unsigned currP = pbeg;
            for (auto i = Base::graph.local_begin (), end_i = Base::graph.local_end ();
              i != end_i; ++i) {

              auto& nd = Base::graph.getData (*i, Galois::MethodFlag::UNPROTECTED);
              GALOIS_ASSERT (nd.partition == -1);

              nd.partition = currP;
              ++size;

              if (size >= Base::partSizeLim) {
                ++currP;
                if (currP >= Base::numPart) {
                  currP = pbeg;
                }
                size = 0;
              }
            }
          });
    }
  };

  struct CyclicPartitioner: public BFSpartitioner {

    using Base = BFSpartitioner;
    
    CyclicPartitioner (
        G& graph,
        M& dagManager,
        PartAdjMatrix& adjMatrix,
        const unsigned numPart)
      : 
        Base (graph, dagManager, adjMatrix, numPart)
    {}

    void partition (void) {
      on_each_impl (
          [this] (const unsigned tid, const unsigned numT) {
            int counter = 0;
            for (auto i = Base::graph.local_begin (), end_i = Base::graph.local_end ();
              i != end_i; ++i) {

              auto& nd = Base::graph.getData (*i, Galois::MethodFlag::UNPROTECTED);
              GALOIS_ASSERT (nd.partition == -1);

              nd.partition = counter % Base::numPart;
              ++counter;
            }
          });
    }
  };


  struct PartMetaData: private boost::noncopyable {

    using LocalWL = Galois::gdeque<GNode, 64>;
    // using LocalWL = typename ContainersWithGAlloc::Deque<GNode>::type;
    using IncomingBoundaryWLmap = std::vector<LocalWL>;

    unsigned id;
    std::atomic<int> indegree;
    IncomingBoundaryWLmap incomingWLs;

    std::vector<PartMetaData*> neighbors;
    LL::SimpleLock mutex;

    LocalWL* currInnerWL = new LocalWL ();
    LocalWL* nextInnerWL = new LocalWL ();
    LocalWL* currBoundaryWL = new LocalWL ();
    LocalWL* nextBoundaryWL = new LocalWL ();
    
    unsigned flips = 0;

    PartMetaData (const unsigned id, const unsigned numPart) 
      : id (id), indegree (0), incomingWLs (numPart) {

    }
    
    ~PartMetaData (void) {
      delete currInnerWL; currInnerWL = nullptr;
      delete nextInnerWL; nextInnerWL = nullptr;
      delete currBoundaryWL; currBoundaryWL = nullptr;
      delete nextBoundaryWL; nextBoundaryWL = nullptr;
    }

    void switchWorkLists (void) {
      assert (currInnerWL->empty ());
      assert (currBoundaryWL->empty ());

      std::swap (currInnerWL, nextInnerWL);
      std::swap (currBoundaryWL, nextBoundaryWL);
    }

    void flipEdges (void) {
      GALOIS_ASSERT (indegree == 0);

      ++flips;
      switchWorkLists ();
      // never increment self indegree and decrement others indegree in the same
      // loop. Increment self indegree first and then others indegree so that if a
      // neighbor becomes source and starts flipping edges, there is no error
      indegree += int (neighbors.size ());


      for (PartMetaData* n: neighbors) {
        int x = --(n->indegree);
        assert (n->indegree >= 0);

        if (x == 0) {
          // std::printf ("partition %d is now a source\n", n->id);
        }
      }

    }

  };

  struct ThreadWorker {

    LL::SimpleLock stealLock;
    Galois::gdeque<PartMetaData*> myPartitions;

    size_t innerIter = 0;
    size_t boundaryIter = 0;


    PartMetaData* takeOne (void) {

      PartMetaData* p = nullptr;
      stealLock.lock();
        if (!myPartitions.empty ()) {
          p = myPartitions.front ();
          myPartitions.pop_front ();
        }
      stealLock.unlock ();

      return p;
    }

    void putOne (PartMetaData* p) {
      GALOIS_ASSERT (p != nullptr);

      stealLock.lock();
          myPartitions.push_back (p);
      stealLock.unlock ();
    }

  };


  struct Ucontext {
    InputGraphPartDAGexecutor& exec;
    PartMetaData& pusher;

    void push (GNode n) {
      exec.push (n, pusher);
    }
    
  };




  static const unsigned PARTITION_MULT_FACTOR = 2;

  G& graph;
  F func;
  M& dagManager;
  const char* loopname;
  TerminationDetection& term;

  unsigned numPart;
  // std::vector<PartMetaData> partitions;
  PartMetaData* partitions;
  PartAdjMatrix adjMatrix;

  InputGraphPartDAGexecutor (G& graph, const F& func, M& dagManager, const char* loopname)
    :
      graph (graph),
      func (func),
      dagManager (dagManager),
      loopname (loopname),
      term(getSystemTermination()), 
      numPart (PARTITION_MULT_FACTOR * Galois::getActiveThreads ())
  {

    // partitions.clear ();
    // for (unsigned i = 0; i < numPart; ++i) {
      // partitions.emplace_back (i);
      // assert (partitions[i].id == i);
    // }

    partitions = reinterpret_cast<PartMetaData*> (malloc (numPart * sizeof (PartMetaData)));
    GALOIS_ASSERT (partitions != nullptr);

    for (unsigned i = 0; i < numPart; ++i) {
      new (partitions + i) PartMetaData (i, numPart);
    }

    adjMatrix.clear ();
    adjMatrix.resize (numPart, std::vector<unsigned> (numPart, 0));
  }


  void initCoarseDAG () {

    for (size_t i = 0; i < adjMatrix.size (); ++i) {
      partitions[i].neighbors.clear ();
      for (size_t j = 0; j < adjMatrix [i].size (); ++j) {
        if (adjMatrix[i][j] != 0) {
          partitions [i].neighbors.push_back (&partitions [j]);
        }
      }
    }

   

    for (unsigned i = 0; i < numPart; ++i) {
      PartMetaData& p = partitions[i];

      p.indegree = 0;

      for (PartMetaData* q: p.neighbors) {
        assert (p.id != q->id);
        if (p.id > q->id) {
          ++(p.indegree);
        }
      }
    }

    // for debug only
    for (unsigned i = 0; i < numPart; ++i) {
      PartMetaData& p = partitions[i];

      if (p.indegree == 0) {
        std::printf ("Partition %d is initially a source\n", i);

        for (PartMetaData* q: p.neighbors) {
          assert (q->indegree != 0);
        }
      }
    }
  }


  void push (GNode n, PartMetaData& pusher) {
    auto& nd = graph.getData (n, Galois::MethodFlag::UNPROTECTED);

    assert (nd.partition != -1);
    PartMetaData& owner = partitions[nd.partition];

    int expected = 0;
    if (nd.onWL.cas (expected, 1)) {

      if (owner.id != pusher.id) {
        // push remote

        GALOIS_ASSERT (nd.isBoundary);

        assert (pusher.id < owner.incomingWLs.size ());
        owner.incomingWLs[pusher.id].push_back (n);

      }
      else {
        // push local
        if (nd.isBoundary) {
          owner.nextBoundaryWL->push_back (n);

        } else {
          owner.nextInnerWL->push_back (n);
        }

      }
    } // end if cas onWL

  }

  // void push (GNode n, PartMetaData& pusher) {
    // auto& nd = graph.getData (n, Galois::MethodFlag::UNPROTECTED);
// 
    // assert (nd.partition != -1);
    // PartMetaData& owner = partitions[nd.partition];
// 
    // int expected = 0;
    // if (nd.onWL.cas (expected, 1)) {
// 
      // if (owner.id != pusher.id) {
        // // push remote
        // 
        // GALOIS_ASSERT (nd.isBoundary);
// 
        // assert (pusher.id < owner.incomingWLs.size ());
        // owner.incomingWLs[pusher.id].push_back (n);
// 
// 
      // } else {
        // // push local
// 
        // if (nd.isBoundary) {
          // owner.currBoundaryWL.push_back (n);
// 
        // } else {
          // owner.currInnerWL.push_back (n);
        // }
      // }
// 
    // } // end if cas
// 
  // }

  template <typename R>
  void fill_initial (const R& range) {

    using LocalContrib = typename Galois::ContainersWithGAlloc::Vector<Galois::gdeque<GNode, 64> >::type;

    Galois::Runtime::on_each_impl (
        [this, &range] (unsigned tid, unsigned numT) {

          LocalContrib localContribInner (numPart);
          LocalContrib localContribBoundary (numPart);

          for (auto i = range.local_begin (), end_i = range.local_end ();
            i != end_i; ++i) {

            auto& nd = graph.getData (*i, Galois::MethodFlag::UNPROTECTED);
            assert (nd.partition != -1);

            if (nd.isBoundary) {
              localContribBoundary [nd.partition].push_back (*i);
            } else {
              localContribInner [nd.partition].push_back (*i);
            }
            
          }

          for (size_t i = 0; i < numPart; ++i) {

            if (!localContribInner [i].empty ()) {
              partitions [i].mutex.lock ();
              for (const auto& n: localContribInner [i]) {
                partitions [i].currInnerWL->push_back (n);
              }
              partitions [i].mutex.unlock ();
            }

            if (!localContribBoundary [i].empty ()) {
              partitions [i].mutex.lock ();

              for (const auto& n: localContribBoundary [i]) {
                partitions [i].currBoundaryWL->push_back (n);
              }
              
              partitions [i].mutex.unlock ();
            }
          }
        });
  }



  // TODO: add iteration stats

  template <typename W>
  void applyOperator (Ucontext& ctxt, W& workList, bool& workHappened, size_t& iter) {

    workHappened = workHappened || !workList.empty ();

    while (!workList.empty ()) {
      ++iter;
      
      GNode n = workList.front ();
      workList.pop_front ();

      auto& nd = graph.getData (n, Galois::MethodFlag::UNPROTECTED);
      assert (nd.partition == ctxt.pusher.id);
      nd.onWL = 0;
      func (n, ctxt);
    }
  }


  template <typename R>
  void execute (const R& range) {
    // 1. partition 
    // 2. create coarsened graph
    // 3. initialize the worklists for partitions
    // 4. work item is a partition, upon choosing a partition, a thread
    //    a. processes currInnerWL work items, until exhausted
    //    b. determines if it is a source and then processes boundary items
    //
    //
    // ISSUES:
    // 1. Determinism?
    // 2. Load-balancing
    //


    BFSpartitioner partitioner (graph, dagManager, adjMatrix, numPart);
    // BlockPartitioner partitioner (graph, dagManager, adjMatrix, numPart);

    partitioner.partition ();

    std::printf ("Graph has %d components\n", partitioner.countComponents ());

    partitioner.initCoarseAdj ();

    initCoarseDAG ();
    
    StatTimer texec ("InputGraphPartDAGexecutor execution time");
    texec.start ();
    fill_initial (range);

    Galois::Runtime::PerThreadStorage<ThreadWorker> workers;

    Galois::Runtime::on_each_impl (
        [this, &workers] (const unsigned tid, const unsigned numT) {
          size_t beg = tid * PARTITION_MULT_FACTOR;
          size_t end = std::min ((tid + 1) * PARTITION_MULT_FACTOR, numPart);

          ThreadWorker& w = *workers.getLocal (tid);
          for (; beg < end; ++beg) {
            w.myPartitions.push_back (&partitions[beg]);
          }
            
          term.initializeThread ();
        });
    

    // TODO: (a) upon picking a partition, either process all worklists
    // and then put it back or process them one at a time, by repeatedly picking the partition. 
    // TODO: order of processing worklists? currInnerWL, local-boundary, external-boundary 
    //
    // TODO: stealing candidates? none, within package only, first within package
    // and then outside
    //
    // TODO: what to do with stolen partition? put back in original owner? keep?

    Galois::Runtime::on_each_impl (
        [this, &workers] (const unsigned tid, const unsigned numT) {

          ThreadWorker& worker = *workers.getLocal (tid);

          while (true) {

            bool workHappened = false; 
            PartMetaData* p = worker.takeOne ();

            if (p != nullptr) {

            // LL::gDebug ("working on parition", p->id);

              Ucontext ctxt{*this, *p};

              applyOperator (ctxt, *(p->currInnerWL), workHappened, worker.innerIter);

              if (p->indegree == 0) {

                applyOperator (ctxt, *(p->currBoundaryWL), workHappened, worker.boundaryIter);

                for (size_t i = 0; i < numPart; ++i) {
                  applyOperator (ctxt, p->incomingWLs[i], workHappened, worker.boundaryIter);
                }

                p->flipEdges ();

              }

            }


            worker.putOne (p);


            term.localTermination (workHappened);
            bool quit = term.globalTermination ();

            if (quit) {
              break;
            }
            
          }

          for (auto i = worker.myPartitions.begin ()
              , end_i = worker.myPartitions.end (); i != end_i; ++i) {
            std::printf ("Partition %d performed %d rounds\n", (*i)->id, (*i)->flips);
          }

          reportStat (loopname, "Inner Iterations", worker.innerIter);
          reportStat (loopname, "Boundary Iterations", worker.boundaryIter);
          reportStat (loopname, "Total Iterations", (worker.innerIter + worker.boundaryIter));


        });

    texec.stop ();

  }


};

template <typename R, typename F, typename G, typename M>
void for_each_det_input_part (const R& range, const F& func, G& graph, M& dagManager, const char* loopname) {

  Galois::Runtime::getSystemThreadPool ().burnPower (Galois::getActiveThreads ());

  InputGraphPartDAGexecutor<G, F, M> executor {graph, func, dagManager, loopname};

  executor.execute (range);

  Galois::Runtime::getSystemThreadPool ().beKind ();

}

// TODO: logic to choose correct DAG type based on graph type or some graph tag
template <typename R, typename G, typename F>
void for_each_det_input_part (const R& range, const F& func, G& graph, const char* loopname) {

  typedef typename DAGmanagerInOut<G>::Manager  M;
  M dagManager {graph};

  for_each_det_input_part (range, func, graph, 
      dagManager, loopname);

}

template <> 
struct ForEachDet_InputDAG<InputDAG_ExecTy::PART> {

  template <typename R, typename F, typename G>
  static void run (const R& range, const F& func, G& graph, const char* loopname) {
    for_each_det_input_part (range, func, graph, loopname);
  }
};



} // end namespace Runtime

} // end namespace Galois

#endif // GALOIS_RUNTIME_DET_PART_INPUT_DAG
