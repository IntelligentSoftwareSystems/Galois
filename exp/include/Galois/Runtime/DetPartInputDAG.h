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

    InputGraphPartDAGexecutor& exec;


    template <typename R, typename B>
    void pickSources (const R& range, const size_t rangeSize, B& sources) {

      ptrdiff_t jump_size = std::max (rangeSize /(2*exec.numPart), size_t (1));

      auto iter = range.begin ();
      size_t dist = 0;

      for (unsigned i = 0; i < exec.numPart && dist < rangeSize; ++i) {

        auto& nd = exec.graph.getData (*iter, Galois::MethodFlag::UNPROTECTED);
        nd.partition = i;
        
        sources.push_back (*iter);
        std::advance (iter, jump_size);
        dist += jump_size;
      }
    }

    template <typename I>
    void bfsTraversal (I beg, I end) {

      using WL = Galois::WorkList::dChunkedFIFO<32>;

      Galois::for_each (beg, end,
          [this] (GNode src, Galois::UserContext<GNode>& ctxt) {
            bool boundary = false;

            auto& sd = exec.graph.getData (src, Galois::MethodFlag::UNPROTECTED);
            assert (sd.partition != -1);

            auto visitAdjClosure = [this, &sd, &ctxt, &boundary] (GNode dst) {
              auto& dd = exec.graph.getData (dst, Galois::MethodFlag::UNPROTECTED);

              if (dd.partition == -1) {
                dd.partition = sd.partition;
                ctxt.push (dst);
              } else if (dd.partition != sd.partition) {
                boundary = true;
              }
            };

            exec.dagManager.applyToAdj (src, visitAdjClosure);

            sd.isBoundary = boundary;
            
          },
          Galois::loopname ("partition_bfs"),
          Galois::wl<WL> ());
    }

    template <typename R, typename W>
    void filterUnpartitioned (const R& range, W& unpartitioned) {

      assert (unpartitioned.empty_all ());
      Galois::Runtime::do_all_coupled (range,
          [this, &unpartitioned] (GNode n) {
            auto& nd = exec.graph.getData (n, Galois::MethodFlag::UNPROTECTED);
            if (nd.partition == -1) {
              unpartitioned.push_back (n);
            }
          },
          "find-unpartitioned");
    }

    void partition (void) {

      Galois::PerThreadBag<GNode, 64>* currRemaining = new PerThreadBag<GNode, 64> ();
      Galois::PerThreadBag<GNode, 64>* nextRemaining = new PerThreadBag<GNode, 64> ();

      Galois::gdeque<GNode, 64> sources;

      bool first = true;
      unsigned rounds = 0;

      do {
        ++rounds;

        sources.clear ();
        if (first) {
          pickSources (makeLocalRange (exec.graph), exec.graph.size (),  sources);

        } else {
          pickSources (makeLocalRange (*currRemaining), currRemaining->size_all (), sources);
        }

        bfsTraversal (sources.begin (), sources.end ());

        if (first) {
          first = false;
          filterUnpartitioned (makeLocalRange (exec.graph), *nextRemaining);

        } else {
          filterUnpartitioned (makeLocalRange (*currRemaining), *nextRemaining);
        }

        std::swap (currRemaining, nextRemaining);
        nextRemaining->clear_all_parallel ();


      } while (!currRemaining->empty_all ());

      std::printf ("BFSpartitioner partitioned in %d rounds\n", rounds);

      delete currRemaining; currRemaining = nullptr;
      delete nextRemaining; nextRemaining = nullptr;

    }

    unsigned countComponents (void) {
      std::vector<unsigned> componentIDs (exec.graph.size (), 0);
      
      auto nextSource = exec.graph.begin ();

      unsigned numComp = 0;

      while (nextSource != exec.graph.end ()) {

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

              exec.dagManager.applyToAdj (n, visitAdjClosure);

            },
            Galois::loopname ("find_component_bfs"),
            Galois::wl<Galois::WorkList::dChunkedFIFO<32> > ());
        

        // find next Source
        for (auto end = exec.graph.end (); nextSource != end; ++nextSource) {
          if (componentIDs[*nextSource] == 0) {
            break;
          }
        }
        
      } // end while

      return numComp;

    }


    void revisitPartitions (void) {
      using ParCounter = Galois::GAccumulator<size_t>;
      using PartCounters = std::vector<ParCounter>;

      PartCounters partCounters (exec.numPart);

      do_all_impl (makeLocalRange (exec.graph),
          [this, &partCounters] (GNode src) {

            bool boundary = false;

            auto& sd = exec.graph.getData (src, Galois::MethodFlag::UNPROTECTED);

            if (sd.partition == -1) {
              unsigned deg = 0;
              auto countDegClosure = [this, &deg, &src] (GNode dst) {
                if (src != dst) {
                  ++deg;
                  std::printf ("src: %d has neighbor %d\n", src, dst);
                }
              };

              exec.dagManager.applyToAdj (src, countDegClosure);

              GALOIS_ASSERT (deg == 0);
            }

            GALOIS_ASSERT (sd.partition != -1);

            partCounters[sd.partition] += 1;

            auto checkPartClosure = [this, &sd, &boundary] (GNode dst) {
              auto& dd = exec.graph.getData (dst, Galois::MethodFlag::UNPROTECTED);

              if (dd.partition != sd.partition) {
                boundary = true;

                GALOIS_ASSERT (dd.partition != -1);
                if (exec.adjMatrix [sd.partition][dd.partition] != 1) {
                  exec.adjMatrix [sd.partition][dd.partition] = 1;
                }

                if (exec.adjMatrix [dd.partition][sd.partition] != 1) {
                  exec.adjMatrix [dd.partition][sd.partition] = 1;
                }
              }
            };


            exec.dagManager.applyToAdj (src, checkPartClosure);

            GALOIS_ASSERT (sd.isBoundary == boundary);
           
          },
          "check_partitions",
          true);

      for (size_t i = 0; i < partCounters.size (); ++i) {
        std::printf ("partition %zd, size =%zd\n", i, partCounters[i].reduceRO ());
      }
      
    }
  }; // end class BFSpartitioner


  struct PartMetaData: private boost::noncopyable {

    using LocalWL = Galois::gdeque<GNode, 1024>;
    using IncomingBoundaryWLmap = std::vector<LocalWL>;

    unsigned id;
    std::atomic<unsigned> indegree;
    IncomingBoundaryWLmap incomingWLs;

    std::vector<PartMetaData*> neighbors;
    LL::SimpleLock mutex;

    LocalWL inner;
    LocalWL boundary;
    

    PartMetaData (const unsigned id, const unsigned numPart) 
      : id (id), indegree (0), incomingWLs (numPart) {

    }

    void flipEdges (void) {
      GALOIS_ASSERT (indegree == 0);

      for (PartMetaData* n: neighbors) {
        --n->indegree;
        assert (n->indegree >= 0);

        ++indegree;
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
    InputGraphPartDAGexecutor& outer;
    PartMetaData& pusher;

    void push (GNode n) {
      outer.push (n, pusher);
    }
    
  };




  static const unsigned PARTITION_MULT_FACTOR = 4;

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


      } else {
        // push local

        if (nd.isBoundary) {
          owner.boundary.push_back (n);

        } else {
          owner.inner.push_back (n);
        }
      }

    } // end if cas

  }

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
                partitions [i].inner.push_back (n);
              }
              partitions [i].mutex.unlock ();
            }

            if (!localContribBoundary [i].empty ()) {
              partitions [i].mutex.lock ();

              for (const auto& n: localContribBoundary [i]) {
                partitions [i].boundary.push_back (n);
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
    //    a. processes inner work items, until exhausted
    //    b. determines if it is a source and then processes boundary items
    //
    //
    // ISSUES:
    // 1. Determinism?
    // 2. Load-balancing
    //


    BFSpartitioner partitioner {*this};

    partitioner.partition ();

    std::printf ("Graph has %d components\n", partitioner.countComponents ());

    partitioner.revisitPartitions ();

    initCoarseDAG ();
    
    fill_initial (range);

    Galois::Runtime::PerThreadStorage<ThreadWorker> workers;

    Galois::Runtime::on_each_impl (
        [this, &workers] (const unsigned tid, const unsigned numT) {
          size_t beg = tid * PARTITION_MULT_FACTOR;
          size_t end = std::max ((tid + 1) * PARTITION_MULT_FACTOR, numPart);

          ThreadWorker& w = *workers.getLocal (tid);
          for (; beg < end; ++beg) {
            w.myPartitions.push_back (&partitions[beg]);
          }
            
          term.initializeThread ();
        });
    

    // TODO: (a) upon picking a partition, either process all worklists
    // and then put it back or process them one at a time, by repeatedly picking the partition. 
    // TODO: order of processing worklists? inner, local-boundary, external-boundary 
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

              Ucontext ctxt{*this, *p};

              applyOperator (ctxt, p->inner, workHappened, worker.innerIter);

              if (p->indegree == 0) {

                applyOperator (ctxt, p->boundary, workHappened, worker.boundaryIter);

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

          reportStat (loopname, "Inner Iterations", worker.innerIter);
          reportStat (loopname, "Boundary Iterations", worker.boundaryIter);
          reportStat (loopname, "Total Iterations", (worker.innerIter + worker.boundaryIter));


        });


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
