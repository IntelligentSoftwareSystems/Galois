
/** Billiards Simulation Order Independence Test -*- C++ -*-
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
 * Billiards Simulation Order Independence Test
 *
 * @author <ahassaan@ices.utexas.edu>
 */


#ifndef _ORDER_DEP_TEST_H_
#define _ORDER_DEP_TEST_H_

#include <cassert>

#include "Event.h"
#include "Ball.h"
#include "Cushion.h"
#include "FPutils.h"

class OrderDepTest {

  static const double V_MAX;

public:
  static bool dependsOn(const Event& later, const Event& earlier) {

    assert (earlier < later);

    return dependsOnInternal (later, earlier);
  }

private:

  static bool dependsOnInternal (const Event& e2, const Event& e1) {
    assert (e1 < e2);

    assert (e1.getBall ().vel ().mag () < V_MAX);
    assert (e2.getBall ().vel ().mag () < V_MAX);

    if (e1.getKind () == Event::BALL_COLLISION) {
      assert (e1.getOtherBall ().vel ().mag () < V_MAX);
    }

    if (e2.getKind () == Event::BALL_COLLISION) {
      assert (e2.getOtherBall ().vel ().mag () < V_MAX);
    }

    bool haveSameBall = checkCommonBall (e1, e2);

    if (haveSameBall) { return true; }

    double minDist = computeMinDist (e1, e2);

    double vmaxTime = minDist / V_MAX;

    double tdiff = e2.getTime () - e1.getTime ();

    assert (tdiff >= 0.0);

    assert (!FPutils::almostEqual (vmaxTime, tdiff));

    return (vmaxTime < tdiff);

  }

  static bool checkCommonBall (const Event& e1, const Event& e2) {
    bool result = false;

    // fast check
    result = isSame (e1.getBall (), e2.getBall ());

    if (!result) {
      // slow check
      if (e1.getKind () == Event::BALL_COLLISION && e2.getKind () == Event::BALL_COLLISION) {

        result = result || isSame (e1.getOtherBall (), e2.getOtherBall ())
          || isSame (e1.getBall (), e2.getOtherBall ())
          || isSame (e1.getOtherBall (), e2.getBall ());

      } else if (e1.getKind () == Event::BALL_COLLISION) {

        result = result || isSame (e1.getOtherBall (), e2.getBall ());

      } else if (e2.getKind () == Event::BALL_COLLISION) {

        result = result || isSame (e1.getBall (), e2.getOtherBall ());

      } else {

        result = result || isSame (e1.getBall (), e2.getBall ());
      }
    }

    return result;
  }

  static bool isSame (const Ball& b1, const Ball& b2) {
    return (b1.getID () == b2.getID ());
  }

  static double computeMinDist (const Event& e1, const Event& e2) {

    double minDist = finalDist (e1, e1.getBall (), e2, e2.getBall ());

    if (e1.getKind () == Event::BALL_COLLISION && e2.getKind () == Event::BALL_COLLISION) {


      double d12 = finalDist (e1, e1.getBall ()     , e2, e2.getOtherBall ());
      double d21 = finalDist (e1, e1.getOtherBall (), e2, e2.getBall ());
      double d22 = finalDist (e1, e1.getOtherBall (), e2, e2.getOtherBall ());


      minDist = std::min (
          std::min (minDist, d22), 
          std::min (d12, d21));

    } else if (e1.getKind () == Event::BALL_COLLISION) {

      double d21 = finalDist (e1, e1.getOtherBall (), e2, e2.getBall ());
      minDist = std::min (minDist, d21);

    } else if (e2.getKind () == Event::BALL_COLLISION) {

      double d12 = finalDist (e1, e1.getBall (), e2, e2.getOtherBall ());
      minDist = std::min (minDist, d12);

    } else {
    }

    return minDist;

  }

  //! distance between ball1 of e1 and ball2 of e2, such
  //! that position of ball1 is evaluated at time of e1
  //! and position of ball2 is evaluated at time of e2
  //! We also subtract sum of radii of ball1 and ball2 
  //! to compute touching distance
  //
  static double finalDist (const Event& e1, const Ball& ball1, const Event& e2, const Ball& ball2) {

    Vec2 ball1Pos = finalPos (e1, ball1);
    Vec2 ball2Pos = finalPos (e2, ball2);

    double sumRadii = (ball1.radius () + ball2.radius ());

    return (ball1Pos.dist (ball2Pos) - sumRadii);
  }

  static Vec2 finalPos (const Event& e, const Ball& ball) {
    assert (isSame (e.getBall (), ball) || isSame (e.getOtherBall (), ball));

    return (ball.pos () + ball.vel () * (e.getTime () - ball.time ()));
  }

};



#endif // _ORDER_DEP_TEST_H_
