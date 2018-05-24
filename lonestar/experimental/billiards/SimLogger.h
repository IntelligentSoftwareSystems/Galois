/** Billiards Logging Utils  -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Billiards Logging Utils.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

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
