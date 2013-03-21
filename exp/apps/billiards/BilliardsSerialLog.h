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

  void updateLog (FILE* simLog, const Ball& b, double time) {
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
      updateLog (simLog, table.getBallByID (i), 0.0);
    }



    PriorityQueue pq;

    for (std::vector<Event>::iterator i = initEvents.begin (), ei = initEvents.end ();
        i != ei; ++i) {

        pq.push (*i);
    }

    size_t iter = 0;
    std::vector<Event> addList;
    // double simTime = 0.0;

    while (!pq.empty ()) {

      Event e = pq.top ();
      pq.pop ();

      if (enablePrints) {
        std::cout << "Processing event=" << e.str () << std::endl;
      }

      // // TODO: remove
      // if (e.getKind () == Event::CUSHION_COLLISION) {
        // std::cout << "Before cushion collision: " << e.getBall ().str () << std::endl;
      // }
      // if (e.getKind () == Event::BALL_COLLISION) {
        // std::cout << "Before ball collision: " << e.getBall ().str ()
          // << "     " << e.getOtherBall ().str () << std::endl;
      // }

      // check staleness before simulating
      const bool notStale = e.notStale ();

      addList.clear ();
      e.simulate (addList, table, endtime);

      // // TODO: remove
      // if (e.getKind () == Event::CUSHION_COLLISION) {
        // std::cout << "After cushion collision: " << e.getBall ().str () << std::endl;
      // }
      // if (e.getKind () == Event::BALL_COLLISION) {
        // std::cout << "After ball collision: " << e.getBall ().str ()
          // << "     " << e.getOtherBall ().str () << std::endl;
      // }

      // may need to add new events for balls in stale events
      for (std::vector<Event>::iterator i = addList.begin (), ei = addList.end ();
          i != ei; ++i) {

        pq.push (*i);

        if (enablePrints) {
          std::cout << "Adding event=" << i->str () << std::endl;
        }
      }

      if (notStale) {
        // update after simulate
        assert ((e.getKind () == Event::BALL_COLLISION || e.getKind () == Event::CUSHION_COLLISION)
            && "unsupported event kind");

        assert (&(e.getBall ()) != NULL);
        updateLog (simLog, e.getBall (), e.getTime ());

        if (e.getKind () == Event::BALL_COLLISION) {

          assert (&(e.getOtherBall ()) != NULL);
          updateLog (simLog, e.getOtherBall (), e.getTime ());
        } 

      } // end if notStale

      if (enablePrints) {
        table.printState (std::cout);
      }

      // // update all the balls to latest simulation time
      // if (e.getTime () > simTime) {
        // simTime = e.getTime ();
        // 
        // table.advance (simTime);
// 
        // table.check ();
      // }


      ++iter;
    }

    fclose (simLog);

    return iter;

  }
};

