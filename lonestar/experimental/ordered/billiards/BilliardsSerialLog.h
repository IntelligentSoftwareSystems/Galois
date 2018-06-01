#include "Billiards.h"
#include "SimLogger.h"

class BilliardsSerialLog: public Billiards {

  typedef std::priority_queue<Event, std::vector<Event>, Event::ReverseComparator> PriorityQueue;
public:

  virtual const std::string version () const { return "Serial Ordered with Event logging"; }

  void printLogHeader (FILE* simLog) {
    fprintf (simLog, "step, time, ball.id, ball.pos.x, ball.pos.y, ball.vel.x, ball.vel.y\n");
  }



  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false) {

    table.writeConfig ();
    table.ballsToCSV ();

    SimLogger simLog;

    PriorityQueue pq;

    for (std::vector<Event>::iterator i = initEvents.begin (), ei = initEvents.end ();
        i != ei; ++i) {

        pq.push (*i);
    }

    size_t iter = 0;
    std::vector<Event> addList;
    // FP simTime = 0.0;

    while (!pq.empty ()) {

      Event e = pq.top ();
      pq.pop ();

      if (enablePrints) {
        std::cout << "Processing event=" << e.str () << std::endl;
      }


      // check staleness before simulating
      const bool notStale = e.notStale ();

      addList.clear ();
      e.simulate();
      table.addNextEvents (e, addList, endtime);

      // may need to add new events for balls in stale events
      for (std::vector<Event>::iterator i = addList.begin (), ei = addList.end ();
          i != ei; ++i) {

        pq.push (*i);

        if (enablePrints) {
          std::cout << "Adding event=" << i->str () << std::endl;
        }
      }

      if (notStale) {
        simLog.log (e);
        simLog.incStep ();
      } // end if notStale

      if (enablePrints) {
        table.printState (std::cout);
      }

      ++iter;
    }

    return iter;

  }
};

