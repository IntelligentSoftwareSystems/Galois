#ifndef BILLIARDS_SIM_LOGGER_H
#define BILLIARDS_SIM_LOGGER_H

#include <cassert>

#include "Billiards.h"

class SimLogger: private boost::noncopyable {

  FILE* logFH;
  unsigned step;

public:
  explicit SimLogger (const char* fileName="simLog.csv"): step (0) {
    assert (fileName != NULL);
    logFH = fopen (fileName, "w");
    assert (logFH != NULL);

    fprintf (logFH, "step, time, ball.id, ball.pos.x, ball.pos.y, ball.vel.x, ball.vel.y\n");
  }

  ~SimLogger () {
    fclose (logFH);
  }

  void log (const Event& e) {
    // update after simulate
    assert ((e.getKind () == Event::BALL_COLLISION || e.getKind () == Event::CUSHION_COLLISION)
        && "unsupported event kind");

    assert (e.getBall () != nullptr);
    logToFile (e.getTime (), e.getBall ());

    if (e.getKind () == Event::BALL_COLLISION) {

      assert (e.getOtherBall () != nullptr);
      logToFile (e.getTime (), e.getOtherBall ());
    } 
  }

  void incStep () { ++step; }

private:

  void logToFile (const FP& time, const Ball* b) {
    assert (FPutils::almostEqual (b.time (), time) && "time stamp mismatch");
    fprintf (logFH, "%d, %e, %d, %e, %e, %e, %e\n", 
        step, time, b->getID (), b->pos ().getX (), b->pos ().getY (), b->vel ().getX (), b->vel ().getY ());
  }
};


#endif // BILLIARDS_SIM_LOGGER_H
