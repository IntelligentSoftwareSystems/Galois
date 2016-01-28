#ifndef BILLIARDS_PARALLEL_H
#define BILLIARDS_PARALLEL_H

#include "Billiards.h"
#include "Galois/PerThreadContainer.h"
#include "Galois/Graphs/Graph.h"


using AddListTy = Galois::PerThreadVector<Event>;

struct VisitNhood {
  static const unsigned CHUNK_SIZE = 1;

  template <typename C, typename I>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& event, const C& c, const I beg, const I end) {

    bool indep = true;

    for (I i = beg; i != end; ++i) {
      if (event > *i) {
        if (OrderDepTest::dependsOn (event, *i)) {
          indep = false;
          break;
        }
      }
    }

    if (!indep) {
      Galois::Runtime::signalConflict ();
    }
  }
};


struct ExecSources {
  static const unsigned CHUNK_SIZE = 4;

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& e) const {
    const_cast<Event&> (e).simulate ();
  }
};

template <typename Tbl_t>
struct AddEvents {

  static const unsigned CHUNK_SIZE = 1;

  Tbl_t& table;
  const FP& endtime;
  AddListTy& addList;
  Accumulator& iter;

  AddEvents (
      Tbl_t& table,
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
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& e, C& ctx) {

    addList.get ().clear ();

    // TODO: use locks to update balls' state atomically 
    // and read atomically
    // const_cast<Event&>(e).simulate ();
    table.addNextEvents (e, addList.get (), endtime);

    for (auto i = addList.get ().begin ()
        , endi = addList.get ().end (); i != endi; ++i) {

      ctx.push (*i);
    }

    iter += 1;
  }
};


#endif // BILLIARDS_PARALLEL_H
