#ifndef _DES_UNORDERED_H_
#define _DES_UNORDERED_H_

#include "Galois/Galois.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/util/Atomic.h"

#include "DESabstractMain.h"

using Galois::AtomicInteger;

class DESunordered: public DESabstractMain {



  struct process {
    Graph& graph;
    std::vector<bool>& onWlFlags;
    AtomicInteger& numEvents;
    AtomicInteger& numIter;

    
    process (
    Graph& graph,
    std::vector<bool>& onWlFlags,
    AtomicInteger& numEvents,
    AtomicInteger& numIter)
      : graph (graph), onWlFlags (onWlFlags), numEvents (numEvents), numIter (numIter) {}




    template <typename ContextTy>
    void operator () (GNode& activeNode, ContextTy& lwl) {
        SimObject* srcObj = graph.getData (activeNode, Galois::Graph::CHECK_CONFLICT);

        // acquire locks on neighborhood: one shot
        for (Graph::neighbor_iterator i = graph.neighbor_begin (activeNode, Galois::Graph::CHECK_CONFLICT)
            , ei = graph.neighbor_end (activeNode, Galois::Graph::CHECK_CONFLICT); i != ei; ++i) {
          // const GNode& dst = *i;
          // SimObject* dstObj = graph.getData (dst, Galois::Graph::CHECK_CONFLICT);
        }



        // should be past the fail-safe point by now


        int proc = srcObj->simulate(graph, activeNode); // number of events processed
        numEvents.addAndGet(proc);

        for (Graph::neighbor_iterator i = graph.neighbor_begin (activeNode, Galois::Graph::NONE)
            , ei = graph.neighbor_end (activeNode, Galois::Graph::NONE); i != ei; ++i) {
          const GNode& dst = *i;

          SimObject* dstObj = graph.getData (dst, Galois::Graph::NONE);

          dstObj->updateActive ();

          if (dstObj->isActive ()) {

            if (!onWlFlags[dstObj->getId ()]) {
              onWlFlags[dstObj->getId ()] = true;
              lwl.push (dst);
            }

          }
        }

        srcObj->updateActive();

        if (srcObj->isActive()) {
          lwl.push (activeNode);
        }
        else {
          onWlFlags[srcObj->getId ()] = false;
        }

        numIter.incrementAndGet();

    }
  };

  /**
   * Run loop.
   *
   */
  virtual void runLoop (const SimInit<Graph, GNode>& simInit) {
    const std::vector<GNode>& initialActive = simInit.getInputNodes();


    std::vector<bool> onWlFlags (simInit.getNumNodes (), false);
    // set onWlFlags for input objects
    for (std::vector<GNode>::const_iterator i = simInit.getInputNodes ().begin (), ei = simInit.getInputNodes ().end ();
        i != ei; ++i) {
      SimObject* srcObj = graph.getData (*i, Galois::Graph::NONE);
      onWlFlags[srcObj->getId ()] = true;
    }



    AtomicInteger numEvents(0);
    AtomicInteger numIter(0);


    process p(graph, onWlFlags, numEvents, numIter);

    Galois::for_each < GaloisRuntime::WorkList::FIFO<GNode> > (initialActive.begin (), initialActive.end (), p);

    std::cout << "Number of events processed = " << numEvents.get () << std::endl;
    std::cout << "Number of iterations performed = " << numIter.get () << std::endl;
  }

  virtual bool isSerial () const { return false; }
};


#endif // _DES_UNORDERED_H_
