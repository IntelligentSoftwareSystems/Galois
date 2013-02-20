#include "Billiards.h"

class BilliardsSerialLog: public Billiards {

  typedef std::priority_queue<Event, std::vector<Event>, Event::ReverseComparator> PriorityQueue;
public:

  virtual const std::string version () const { return "Serial Ordered with Event logging"; }

  void writeConfig (const Table& table, const std::string& confName="config.csv") {
    // TODO:
    FILE* confFile = fopen (confName.c_str (), "w");
    assert (confFile != NULL);

    fprintf (confFile, "length, width, num_balls, ball.mass, ball.radius\n");
    fprintf (confFile, "%e, %e, %d, %e, %e\n",
        table.getLength (), table.getWidth (), table.getNumBalls (), Table::DefaultValues::BALL_MASS, Table::DefaultValues::BALL_RADIUS);

    fclose (confFile);
  }

  void printLogHeader (FILE* simLog) {
    fprintf (simLog, "ball.id, time, ball.pos.x, ball.pos.y, ball.vel.x, ball.vel.y\n");
  }

  void updateBall (FILE* simLog, const Ball& b, double time) {
    assert (FPutils::almostEqual (b.time (), time) && "time stamp mismatch");
    fprintf (simLog, "%d, %e, %e, %e, %e, %e\n", 
        b.getID (), time, b.pos ().getX (), b.pos ().getY (), b.vel ().getX (), b.vel ().getY ());
  }


  virtual size_t runSim (Table& table, std::vector<Event>& initEvents, const double endtime, bool enablePrints=false) {

    writeConfig (table);

    FILE* simLog = fopen ("simLog.csv", "w");
    assert (simLog != NULL);

    printLogHeader (simLog);
    
    for (unsigned i = 0; i < table.getNumBalls (); ++i) {
      updateBall (simLog, table.getBallByID (i), 0.0);
    }



    PriorityQueue pq;

    for (std::vector<Event>::iterator i = initEvents.begin (), ei = initEvents.end ();
        i != ei; ++i) {

        pq.push (*i);
    }

    size_t iter = 0;
    std::vector<Event> addList;

    while (!pq.empty ()) {

      Event e = pq.top ();
      pq.pop ();

      if (enablePrints) {
        std::cout << "Processing event=" << e.str () << std::endl;
      }

      addList.clear ();
      e.simulate (addList, table, endtime);

      // may need to add new events for balls in stale events
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

      if (e.notStale ()) {
        // update after simulate
        if (e.getKind () == Event::CUSHION_COLLISION) {
          assert (&(e.getBall ()) != NULL);
          updateBall (simLog, e.getBall (), e.getTime ());

        } else if (e.getKind () == Event::BALL_COLLISION) {

          assert (&(e.getOtherBall ()) != NULL);
          updateBall (simLog, e.getOtherBall (), e.getTime ());

        } else {
          assert (false && "unsupported event");
          abort ();
        }

      } // end if notStale

      ++iter;
    }

    fclose (simLog);

    return iter;

  }
};

