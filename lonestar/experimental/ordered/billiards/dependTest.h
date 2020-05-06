
/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef _ORDER_DEP_TEST_H_
#define _ORDER_DEP_TEST_H_

#include "Event.h"
#include "Ball.h"
#include "Cushion.h"
#include "FPutils.h"
#include "DefaultValues.h"

#include "llvm/Support/CommandLine.h"

#include <cassert>

namespace cll = llvm::cl;
extern cll::opt<unsigned> vmaxFactor;

class OrderDepTest {

  const FP V_MAX;

public:
  OrderDepTest(void)
      : V_MAX(DefaultValues::MAX_SPEED * FP(unsigned(vmaxFactor))) {}

  bool dependsOn(const Event& later, const Event& earlier) const {

    assert(earlier < later);

    return dependsOnInternal(later, earlier);
  }

private:
  bool dependsOnInternal(const Event& e2, const Event& e1) const {
    assert(e1 < e2);

    GALOIS_ASSERT(e1.getBall()->vel().mag() < V_MAX);
    GALOIS_ASSERT(e2.getBall()->vel().mag() < V_MAX);

    if (e1.getKind() == Event::BALL_COLLISION) {
      GALOIS_ASSERT(e1.getOtherBall()->vel().mag() < V_MAX);
    }

    if (e2.getKind() == Event::BALL_COLLISION) {
      GALOIS_ASSERT(e2.getOtherBall()->vel().mag() < V_MAX);
    }

    bool haveSameBall = checkCommonBall(e1, e2);

    if (haveSameBall) {
      return true;
    }

    FP minDist = computeMinDist(e1, e2);

    assert(V_MAX > FP(0.0));
    FP vmaxTime = minDist / V_MAX;

    // std::cout << "V_MAX = " << V_MAX << std::endl;

    FP tdiff = e2.getTime() - e1.getTime();

    assert(tdiff >= FP(0.0));

    assert(!FPutils::almostEqual(vmaxTime, tdiff));

    return (vmaxTime < tdiff);
  }

  static bool checkCommonBall(const Event& e1, const Event& e2) {
    bool result = false;

    // fast check
    result = isSame(e1.getBall(), e2.getBall());

    if (!result) {
      // slow check
      if (e1.getKind() == Event::BALL_COLLISION &&
          e2.getKind() == Event::BALL_COLLISION) {

        result = result || isSame(e1.getOtherBall(), e2.getOtherBall()) ||
                 isSame(e1.getBall(), e2.getOtherBall()) ||
                 isSame(e1.getOtherBall(), e2.getBall());

      } else if (e1.getKind() == Event::BALL_COLLISION) {

        result = result || isSame(e1.getOtherBall(), e2.getBall());

      } else if (e2.getKind() == Event::BALL_COLLISION) {

        result = result || isSame(e1.getBall(), e2.getOtherBall());

      } else {

        result = result || isSame(e1.getBall(), e2.getBall());
      }
    }

    return result;
  }

  static bool isSame(const Ball* b1, const Ball* b2) {
    return (b1->getID() == b2->getID());
  }

  static FP computeMinDist(const Event& e1, const Event& e2) {

    FP minDist = finalDist(e1, e1.getBall(), e2, e2.getBall());

    if (e1.getKind() == Event::BALL_COLLISION &&
        e2.getKind() == Event::BALL_COLLISION) {

      FP d12 = finalDist(e1, e1.getBall(), e2, e2.getOtherBall());
      FP d21 = finalDist(e1, e1.getOtherBall(), e2, e2.getBall());
      FP d22 = finalDist(e1, e1.getOtherBall(), e2, e2.getOtherBall());

      minDist = std::min(std::min(minDist, d22), std::min(d12, d21));

    } else if (e1.getKind() == Event::BALL_COLLISION) {

      FP d21  = finalDist(e1, e1.getOtherBall(), e2, e2.getBall());
      minDist = std::min(minDist, d21);

    } else if (e2.getKind() == Event::BALL_COLLISION) {

      FP d12  = finalDist(e1, e1.getBall(), e2, e2.getOtherBall());
      minDist = std::min(minDist, d12);

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
  static FP finalDist(const Event& e1, const Ball* ball1, const Event& e2,
                      const Ball* ball2) {

    Vec2 ball1Pos = finalPos(e1, ball1);
    Vec2 ball2Pos = finalPos(e2, ball2);

    FP sumRadii = (ball1->radius() + ball2->radius());

    return (ball1Pos.dist(ball2Pos) - sumRadii);
  }

  static Vec2 finalPos(const Event& e, const Ball* ball) {
    assert(isSame(e.getBall(), ball) || isSame(e.getOtherBall(), ball));

    return (ball->pos() + ball->vel() * (e.getTime() - ball->time()));
  }
};

#endif // _ORDER_DEP_TEST_H_
