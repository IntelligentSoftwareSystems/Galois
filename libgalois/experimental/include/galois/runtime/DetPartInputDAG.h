/**  -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 */

#ifndef GALOIS_RUNTIME_DET_PART_INPUT_DAG
#define GALOIS_RUNTIME_DET_PART_INPUT_DAG

#include "galois/AltBag.h"

#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/Termination.h"
#include "galois/runtime/DetChromatic.h"
#include "galois/runtime/DoAllCoupled.h"
#include "galois/runtime/DetPartitioners.h"

#include <atomic>
#include <functional>


namespace galois {
namespace runtime {

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
  using Bag_ty = galois::PerThreadBag<GNode>;
  using PartAdjMatrix = std::vector<std::vector<unsigned> >;

  struct PartMetaData: private boost::noncopyable {

    using LocalWL = galois::gdeque<GNode, 64>;
    // using LocalWL = typename gstl::Deque<GNode>;
    using IncomingBoundaryWLmap = std::vector<LocalWL>;

    unsigned id;
    std::atomic<int> indegree;
    IncomingBoundaryWLmap incomingWLs;

    std::vector<PartMetaData*> neighbors;
    substrate::SimpleLock mutex;

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

    substrate::SimpleLock stealLock;
    galois::gdeque<PartMetaData*> myPartitions;

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




  static const unsigned PARTITION_MULT_FACTOR = 4;

  G& graph;
  F func;
  M& dagManager;
  const char* loopname;
  substrate::TerminationDetection& term;

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
      term(substrate::getSystemTermination(galois::getActiveThreads())),
      numPart (PARTITION_MULT_FACTOR * galois::getActiveThreads ())
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
    auto& nd = graph.getData (n, galois::MethodFlag::UNPROTECTED);

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
    // auto& nd = graph.getData (n, galois::MethodFlag::UNPROTECTED);
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

    using LocalContrib = typename galois::gstl::Vector<galois::gdeque<GNode, 64> >;

    galois::on_each(
        [this, &range] (unsigned tid, unsigned numT) {

          LocalContrib localContribInner (numPart);
          LocalContrib localContribBoundary (numPart);

          for (auto i = range.local_begin (), end_i = range.local_end ();
            i != end_i; ++i) {

            auto& nd = graph.getData (*i, galois::MethodFlag::UNPROTECTED);
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

      auto& nd = graph.getData (n, galois::MethodFlag::UNPROTECTED);
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


    GreedyPartitioner<G, M> partitioner (graph, dagManager, numPart);
    // CyclicPartitioner<G, M> partitioner (graph, dagManager, numPart);

    partitioner.partition ();

    // std::printf ("Graph has %d components\n", partitioner.countComponents ());

    partitioner.initCoarseAdj (adjMatrix);

    initCoarseDAG ();

    StatTimer texec ("InputGraphPartDAGexecutor execution time");
    texec.start ();
    fill_initial (range);

    galois::substrate::PerThreadStorage<ThreadWorker> workers;

    galois::on_each(
        [this, &workers] (const unsigned tid, const unsigned numT) {
          ThreadWorker& w = *workers.getLocal (tid);

          // // block assignment
          // size_t beg = tid * PARTITION_MULT_FACTOR;
          // size_t end = std::min ((tid + 1) * PARTITION_MULT_FACTOR, numPart);
//
          // for (; beg < end; ++beg) {
            // w.myPartitions.push_back (&partitions[beg]);
          // }

          // cyclic assignment;
          for (unsigned i = tid; i < numPart; i += numT) {
            w.myPartitions.push_back (&partitions[i]);
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

    galois::on_each(
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

          reportStat_Tsum (loopname, "Inner Iterations", worker.innerIter);
          reportStat_Tsum (loopname, "Boundary Iterations", worker.boundaryIter);
          reportStat_Tsum (loopname, "Total Iterations", (worker.innerIter + worker.boundaryIter));


        });

    texec.stop ();

  }


};

template <typename R, typename F, typename G, typename M>
void for_each_det_input_part (const R& range, const F& func, G& graph, M& dagManager, const char* loopname) {

  galois::substrate::getThreadPool ().burnPower (galois::getActiveThreads ());

  InputGraphPartDAGexecutor<G, F, M> executor {graph, func, dagManager, loopname};

  executor.execute (range);

  galois::substrate::getThreadPool ().beKind ();

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



} // end namespace runtime

} // end namespace galois

#endif // GALOIS_RUNTIME_DET_PART_INPUT_DAG
