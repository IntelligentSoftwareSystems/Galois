#include "Billiards.h"
#include "dependTest.h"
#include "BilliardsParallel.h"
#include "BilliardsParallelSectored.h"

#include "galois/runtime/KDGtwoPhase.h"

class BilliardsSectoredIKDG: public Billiards<BilliardsSectoredIKDG>  {

public:

  virtual const std::string version () const { return "IKDG with custom safety test"; }

  template <typename Tbl_t>
  size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false, bool logEvents=false) {

    AddListTy addList;
    Accumulator iter;

    galois::runtime::for_each_ordered_ikdg_custom_safety (
        galois::runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        Event::Comparator (),
        SectorLocalTest<Tbl_t> {table},
        ExecSources (),
        AddEvents<Tbl_t> (table, endtime, addList, iter),
        "sectored-ikdg-custom-safety");

    return iter.reduce ();

  }
};

int main (int argc, char* argv[]) {
  BilliardsSectoredIKDG s;
  s.run (argc, argv);
  return 0;
}
