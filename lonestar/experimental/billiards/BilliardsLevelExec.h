#ifndef BILLIARDS_LEVEL_EXEC_H
#define BILLIARDS_LEVEL_EXEC_H

#include "galois/graphs/Graph.h"
#include "galois/PerThreadContainer.h"

#include "galois/runtime/LevelExecutor.h"

#include "Billiards.h"

class BilliardsLevelExec: public Billiards<BilliardsLevelExec> {

public:

  using Graph = galois::graphs::FirstGraph<void*, void, true>;
  using GNode = Graph::GraphNode;
  using VecNodes = std::vector<GNode>;
  using AddListTy = galois::PerThreadVector<Event>;

  struct GetEventTime {
    const FP& operator () (const Event& e) const { 
      return e.getTime ();
    }
  };

  struct OpFunc {

    static const unsigned CHUNK_SIZE = 1;

    Table& table;
    const FP& endtime;
    AddListTy& addList;
    Accumulator& iter;

    OpFunc (
        Table& table,
        const FP& endtime,
        AddListTy& addList,
        Accumulator& iter)
      :
        table (table),
        endtime (endtime),
        addList (addList),
        iter (iter)
    {}


    template <typename C>
    void operator () (const Event& e, C& ctx) const {

      addList.get ().clear ();

      // TODO: use locks to update balls' state atomically 
      // and read atomically
      const_cast<Event&>(e).simulate ();
      table.addNextEvents (e, addList.get (), endtime);

      for (auto i = addList.get ().begin ()
          , endi = addList.get ().end (); i != endi; ++i) {

        ctx.push (*i);
      }

      iter += 1;
    }
  };

public:

  virtual const std::string version () const { return "using Level-by-Level Executor"; }

  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false) {

    Graph graph;
    VecNodes nodes;

    AddListTy addList;
    Accumulator iter;

    createLocks (table, graph, nodes);

    galois::runtime::for_each_ordered_level (
        galois::runtime::makeStandardRange (initEvents.begin (), initEvents.end ()),
        GetEventTime (), std::less<FP> (),
        VisitNhood<Graph, VecNodes> (graph, nodes),
        OpFunc (table, endtime, addList, iter));

    return iter.reduce ();

  }

};


#endif // BILLIARDS_LEVEL_EXEC_H
