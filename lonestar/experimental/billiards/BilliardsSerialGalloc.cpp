#include "Billiards.h"

#include "Galois/PriorityQueue.h"

template <typename Tbl_t=Table<Ball> >
class BilliardsSerialGalloc: public Billiards<BilliardsSerialGalloc<Tbl_t>, Tbl_t> {

  // using PriorityQueue =  std::priority_queue<Event, std::vector<Event>, Event::ReverseComparator>; 
  using PriorityQueue =  Galois::ThreadSafeOrderedSet<Event, Event::Comparator>;

public:

  virtual const std::string version () const { return "Serial Ordered with Priority Queue"; }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void processEvent (Event& e, Tbl_t& table, std::vector<Event>& addList, const FP& endtime) {

      addList.clear ();
      e.simulate ();
      table.addNextEvents (e, addList, endtime);
  }

  size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false, bool logEvents=false) {

    std::printf ("BilliardsSerialGalloc: number of initial events: %zd\n", initEvents.size ());

    PriorityQueue pq;

    for (std::vector<Event>::iterator i = initEvents.begin (), ei = initEvents.end ();
        i != ei; ++i) {

        pq.push (*i);
    }

    size_t iter = 0;
    std::vector<Event> addList;

    while (!pq.empty ()) {

      Event e = pq.pop ();

      if (enablePrints) {
        std::cout << "Processing event=" << e.str () << std::endl;
      }

      processEvent (e, table, addList, endtime);

      if (logEvents) {
        table.logCollisionEvent (e);
      }

      for (std::vector<Event>::iterator i = addList.begin (), ei = addList.end ();
          i != ei; ++i) {

        pq.push (*i);

        if (enablePrints) {
          std::cout << "Adding event=" << i->str () << std::endl;
        }
      }

      if (enablePrints) {
        table.printState (std::cout);
      }

      ++iter;
    }

    return iter;

  }
};

int main (int argc, char* argv[]) {
  BilliardsSerialGalloc<> s;
  s.run (argc, argv);
  return 0;
}
