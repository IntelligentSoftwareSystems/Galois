#include "galois/runtime/KDGtwoPhase.h"

#include "Billiards.h"
#include "dependTest.h"
#include "BilliardsParallel.h"

class BilliardsTwoPhase: public Billiards<BilliardsTwoPhase, Table<Ball> >  {

public:

  using Tbl_t = Table<Ball>;

  virtual const std::string version () const { return "using IKDG"; }

  size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false, bool logEvents=false) {

    AddListTy addList;
    Accumulator iter;

    galois::runtime::for_each_ordered_ikdg (
        galois::runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        Event::Comparator (),
        VisitNhoodSafetyTest (),
        ExecSources (),
        AddEvents<Tbl_t> (table, endtime, addList, iter, enablePrints), 
        std::make_tuple (galois::loopname ("billiards-ikdg")));

    return iter.reduce ();

  }
};

int main (int argc, char* argv[]) {
  BilliardsTwoPhase s;
  s.run (argc, argv);
  return 0;
}
