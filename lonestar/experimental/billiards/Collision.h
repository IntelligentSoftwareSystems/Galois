#ifndef _COLLISION_H_
#define _COLLISION_H_

#include "FPutils.h"
#include "Ball.h"
#include "Cushion.h"
#include "Event.h"


#include "galois/optional.h"




#include <string>
#include <iostream>
#include <iterator>

#include <boost/noncopyable.hpp>

#include <cstdlib>
#include <cstdio>
#include <cmath>

class Collision: boost::noncopyable {

public:

  //! @param b1 Ball
  //! @param b2 Ball
  //! @return a pair <bool, FP>, 
  //! first value is true if collision
  //! will happen, 2nd value is the time when the collision should
  //! happen
  //!
  //! Basically, we extend the line of trajectory of each ball using its current
  //! pos and vel. If the two lines intersect or ever come closer than
  //! (b1.radius () + b2.radius ()), then a collision will happen, else it won't
  //! happen
  //!
  //! Dist { (b1.pos () + dt1*b1.vel ()), (b2.pos () + dt2*b2.vel ()) } <= (b1.radius () + b2.radius ())
  //! here dt1 = t - b1.time (); dt2 = t - b2.time (), t being the time of collision
  //!
  //! => mag ( (b1.pos () + 1t*b1.vel ()) - (b2.pos () + dt2*b2.vel ()) ) <= (b1.radius () + b2.radius ())
  //!
  //! Solving the resulting quadratic equation gives us the result


  static std::pair<bool, FP> computeCollisionTime (const Ball& ball1, const Ball& ball2) {

    Vec2    b1pos =  ball1.pos ();
    Vec2    b1vel =  ball1.vel ();
    FP  b1time = ball1.time ();
    FP  b1rad =  ball1.radius ();

    Vec2    b2pos =  ball2.pos ();
    Vec2    b2vel =  ball2.vel ();
    FP  b2time = ball2.time ();
    FP  b2rad =  ball2.radius ();

    Vec2 diffV = b1vel - b2vel;

    if (FPutils::almostEqual(diffV.magSqrd (), 0.0)) {
      return std::make_pair (false, -1.0);
    }

    // D =  pos () - time ()*vel ()
    // diffD = D1 - D2
    Vec2 diffD =  (b1pos - b1vel * b1time) - (b2pos - b2vel * b2time);

    // if (diffD.mag () < (b1rad + b2rad)) {
      // std::printf ("diffD.magSqrd (): %g\n", double (diffD.magSqrd ()));
    // }
    // assert (diffD.mag () >= (b1rad + b2rad));

    FP sumRadius = (b1rad + b2rad);
    FP sumRadiusSqrd =  sumRadius * sumRadius;

    FP diffVdiffD = diffV.dot (diffD);

    // Let discr be the term under the square root (discriminant) in quadratic
    // formula = 1/2a * (-b +- sqrt (b^2 - 4ac))
    // actually discr is the actual discriminant divided by 4

    // Let diffV = (x1,y1)
    // and diffD = (x2,y2)
    //
    // discr = b^2 - 4ac
    //    = (diffV.diffD)^2 - 4 (diffV) (diffD - sumRadiusSqrd)
    //
    //    simplifying the vector expression using (x1,y1) & (x2,y2) instead 
    //    of diffV and diffD, we get:
    // discr = (x1y2 - x2y1)^2 - sumRadiusSqrd (x1^2 + y1^2);

    // FP discr = (diffVdiffD * diffVdiffD) - (diffV.magSqrd ()) * (diffD.magSqrd () - sumRadiusSqrd);
    
    FP x1 = diffV.getX ();
    FP y1 = diffV.getY ();

    FP x2 = diffD.getX ();
    FP y2 = diffD.getY ();

    FP discr = sumRadiusSqrd * (diffV.magSqrd ()) - (x1*y2 - x2*y1) * (x1*y2 - x2*y1);

    // std::printf ("discr = %10.10lf,  d2=%10.10lf\n", double (discr), double (d2));

    if (discr > FP (0.0)) {
      // solution is real
      FP t1 = (-diffVdiffD - FPutils::sqrt (discr)) / diffV.magSqrd ();
      FP t2 = (-diffVdiffD + FPutils::sqrt (discr)) / diffV.magSqrd ();

      FP t = -1.0;
      bool valid = false;

      // check if t1 is valid for both
      if (isInFuture (b1time, t1) && isInFuture (b2time, t1)) {
        t = t1;
        valid = true;

      } else {
      }

      // check if t2 is valid for both
      // if valid is already true then pick the minimum time
      if (isInFuture (b1time, t2) && isInFuture (b2time, t2)) {
        t = (valid) ? std::min (t1, t2): t2;
        valid = true;

      } else {
      }

      if (valid) {
        valid = isApproaching (ball1, ball2, t);
      }

      return std::make_pair (valid, t);

    } else {
      return std::make_pair (false, -1.0);
    }

  }


  //! Cushion is a line segment with two end points 'start' and 'end'
  //! Equation is: r(end) + (1-r)start, where 0 <= r <= 1
  //! We compute another parallel to the cushion towards the inside
  //! of the table, which is at a distance R from cushion, R being radius ()
  //! of the ball.
  //! Then we compute the time when the ball will cross this imaginary 
  //! line.
  //! To compute this line, we assume that cushion length vector
  //! 'end-start' goes clockwise around the table. The right unit normal
  //! 'u' always points inwards. We scale this right unit normal by R, 
  //! and get the equation for crossing line
  //! r(end + R*u) + (1 - r)(start + R*u)
  //! = r(end) + (1-r)(start) + R*u
  //! 
  //! We now compute the ratio 'r' and time 't' when ball crosses
  //! the imaginary line.
  //! Position at time 't' = b.pos () + (t - b.time ()) * b.vel ()
  //!  == r(end) + (1-r)(start) + R*u
  //!  Split the above equation into x and y components
  //!  and solve for 'r' and 't'
  //! 


  static std::pair<bool, FP> computeCollisionTime (const Ball& ball, const Cushion& cush) {
    return computeCollisionTime (ball, cush.getLineSegment ());
  }

  static std::pair<bool, FP> computeCollisionTime (const Ball& ball, const LineSegment& lineSeg, const bool useGhostPos=false) {

    Vec2 L = lineSeg.lengthVec ();

    Vec2    bvel =  ball.vel ();
    FP  brad =  ball.radius ();

    Vec2 bpos = ball.pos  (); 
    FP btime = ball.time ();
    
    if (useGhostPos) {
      bpos = ball.ghostPos ();
      btime = ball.ghostTime ();
    }

    FP denominator = (bvel.getX () * L.getY ()) - (bvel.getY () * L.getX ());

    if (FP::fabs(denominator) < FPutils::EPSILON) {
      return std::make_pair (false, -1.0);
    }

    Vec2 D = bpos - bvel * btime;

    Vec2 R = L.rightNormal ().unit ();
    R *= brad;

    Vec2 common = D - lineSeg.getBegin () - R;

    FP r = (bvel.getX () * common.getY ()) - (bvel.getY () * common.getX ());
    r /= denominator;

    FP t = (L.getX () * common.getY ()) - (L.getY () * common.getX ());
    t /= denominator;


    if ((r < FP (0.0)) || (r > FP (1.0)) || (!isInFuture(btime, t))) {
      return std::make_pair (false, -1.0);

    } else {
      assert (isInFuture (btime, t) && "Collision time in the past?");

      bool app = isApproaching (ball, lineSeg);

      return std::make_pair (app, t);

    }

  }



  //! 
  //! We assume that collision with the cushion follows laws of
  //! reflection. We compute the component of balls veclocity
  //! along the cushion length unit vector and the component 
  //! along right normal unit vector. 
  //!
  //! Let v_b and v_a be velocities before and after the collision
  //! Let L be the length vector of the cushion
  //! By laws of reflection
  //! v_b.L == v_a.L
  //!
  //! and, let P be right normal of the cushion
  //!
  //! v_b.P == - v_a.P 
  //!
  //!
  //!
  //!
  //! These two equations
  //! can then be used to compute x and y components of 
  //! resulting vel. The resulting vel is also
  //! multiplied by Cushions REFLECTION_COEFF, in order
  //! to model the slowing down of balls after hitting the cushion etc.
  //!
  static void simulateCollision (Ball& ball, Cushion& cush, const FP& time) {

    Vec2 L = cush.getLineSegment ().lengthVec ();

    Vec2 bvel = ball.vel ();

    FP t = time;

    // component along the length
    FP tangent = L.dot (bvel);

    // component along the right normal
    FP normal = L.rightNormal ().dot (bvel);

    FP denom = L.magSqrd ();

    // x component of vel after collision
    FP v_a_x = ((L.getX () * tangent) - (L.getY () * normal)) / denom;

    // y component of vel after collision
    FP v_a_y = ((L.getY () * tangent) + (L.getX () * normal)) / denom;

    Vec2 v_a (v_a_x, v_a_y);

    FPutils::checkError (ball.mom (ball.vel ()).mag (), ball.mom (v_a).mag ());
    FPutils::checkError (ball.ke (ball.vel ()), ball.ke (v_a));

    ball.update (v_a, t);

  }


  //! Simulate the collision of two balls at time t
  //!
  //! We do so by computing updated positions of b1 and b2
  //! The direction of force applied by either ball on the other
  //! is determined by the vector between the centers of the balls
  //! when they collide. e.g. a moving ball collides with a stationary ball,
  //! the stationary ball will move in the direction determined by
  //! the vector between their centers at the point of contact.
  //!
  //! Let C = b1.newPos - b2.newPos
  //!
  //! Thus we break each balls vel into two components,
  //! a component along C and a component perpendicular to C
  //! The perpendicular component N is unchanged
  //! For components along C, we apply elastic collision of
  //! masses in 1-D perserving momentum and kinetic energy
  //! and compute resulting velocities of b1 and b2 along C.
  //! Note, that direction of C (b1-b2) or (b2-b1) does not matter
  //!
  //!
  //!
  //!
  static void simulateCollision (Ball& ball1, Ball& ball2, const FP& time) {

    Vec2    b1pos =  ball1.pos ();
    Vec2    b1vel =  ball1.vel ();
    FP  b1time = ball1.time ();
    FP  b1mass = ball1.mass ();

    Vec2    b2pos =  ball2.pos ();
    Vec2    b2vel =  ball2.vel ();
    FP  b2time = ball2.time ();
    FP  b2mass = ball2.mass ();

    FP t = time;


    assert (t >= b1time && t >= b2time);

    b1pos = ball1.pos (t);
    b2pos = ball2.pos (t);

    Vec2 C = b2pos - b1pos;

    Vec2 N = C.rightNormal (); // shouldn't matter, left or right normal is fine


    FP sumMass = b1mass + b2mass;

    // Magnitude of component vel along C 
    FP v1_tangent_before = b1vel.dot (C);
    FP v2_tangent_before = b2vel.dot (C);

    FP v1_tangent_after = ((b1mass - b2mass) * v1_tangent_before  +  (FP (2.0) * b2mass) * v2_tangent_before) / sumMass;

    FP v2_tangent_after = ((b2mass - b1mass) * v2_tangent_before  +  (FP (2.0) * b1mass) * v1_tangent_before) / sumMass;


    Vec2 V1_tangent_after = C * (v1_tangent_after / C.magSqrd ()); 
    Vec2 V2_tangent_after = C * (v2_tangent_after / C.magSqrd ());

    // Magnitude of component normal to C. Remains the same before and
    // after
    Vec2 V1_normal_after = N * (ball1.vel ().dot (N) / N.magSqrd ());

    Vec2 V2_normal_after = N * (ball2.vel ().dot (N) / N.magSqrd ());

    Vec2 V1_after = V1_tangent_after + V1_normal_after;
    Vec2 V2_after = V2_tangent_after + V2_normal_after;



    // momentum and kinetic energy should be preserved
    // i.e. should be the same before and after the collision

    Vec2 momBefore = (ball1.mom (ball1.vel ()) + ball2.mom (ball2.vel ()));
    Vec2 momAfter = (ball1.mom (V1_after) + ball2.mom (V2_after));

    FPutils::checkError (momBefore, momAfter);

    FP keBefore = (ball1.ke (ball1.vel ()) + ball2.ke (ball2.vel ()));
    FP keAfter = (ball1.ke (V1_after) + ball2.ke (V2_after));

    FPutils::checkError (keBefore, keAfter);

    ball1.update (V1_after, t);
    ball2.update (V2_after, t);

  }

  static bool isValidCollision (const Ball* b1, const Ball* b2, const FP& time) {
    assert (b1);
    assert (b2);

    Vec2 pos1 = b1->pos (time);
    Vec2 pos2 = b2->pos (time);

    if (FPutils::almostEqual (pos1.dist (pos2), b1->radius () + b2->radius ())) {
      return true;
    }

    return false;
  }

  static bool isValidCollision (const Ball* b1, const Cushion* c, const FP& time) {
    assert (b1);
    assert (c);

    Vec2 p = b1->pos (time);

    if (FPutils::almostEqual (c->getLineSegment ().distanceFrom (p), b1->radius ())) {
      return true;
    }

    return false;


  }

  static bool isValidCollision (const Event& e) {
    if (e.getKind () == Event::BALL_COLLISION) {
      return isValidCollision (e.getBall (), e.getOtherBall (), e.getTime ());

    } else if (e.getKind () == Event::CUSHION_COLLISION) {
      return isValidCollision (e.getBall (), e.getCushion (), e.getTime ());

    } else {
      std::abort ();
      return false;
    }
  }

private:

  //! The time for collision is estimated by extending the lines of
  //! trajectory of each ball and compute the point of intersection with
  //! the trajector of the other ball or the cushion. This point of intersection
  //! may lie in the direction of the ball, or exactly opposite. When the point of
  //! intersection is opposite to the direction of velocity of a ball, then usually
  //! that means the collision time is in the past of the ball and it's not a valid 
  //! collision. Also, dot product of the distance vector (between current position
  //! and position at the time of collision) and velocity should be negative i.e. both are 
  //! in opposite directions. We show below that if this dot product is negative, then the collision is invalid and
  //! must be in the past of the ball.
  //! 
  //! Let 
  //! Vec2 newPos = b.pos + (collisionTime - b.time) * b.vel
  //! Vec2 dist = newPos - b.pos = (collisionTime - b.time) * b.vel
  //! dist.vel = (collisionTime - b.time) * (b.vel . b.vel)
  //!
  //! We know that for vectors v.v > 0.0
  //! thus if dist.vel < 0.0 then it must be the case that (collisionTime < b.time)


  static bool isInFuture (const FP& ballTime, const FP& collisionTime) {
    assert (ballTime >= FP (0.0));
    // assert (ballTime != collisionTime);
    return (collisionTime >= ballTime || FPutils::almostEqual (collisionTime, ballTime));
  }


  // If a ball b has just collided with cushion c (or another ball b2)
  // the equations in computeCollisionTime will show that ball b can collid e with c i.e.
  // repeat the same collision that has just taken place
  // To avoid such repetitions, we check if the ball is moving towards 
  // or way from the cushion. This can be done by dot product of velocity
  // and inward (right) normal of cushion. If dot product is -ve, then ball
  // is moving towards the cushion, else it's moving away from it.
  //

  static bool isApproaching (const Ball& b, const LineSegment& l) {
    // assuming that t is a valid collision time

    FP d = b.vel ().dot (l.lengthVec ().rightNormal ());
    return (d < FP (0.0));
  }


  // To determine, if two balls b1 and b2 are approaching each other, 
  // we compute their positions at collision time t. The balls should 
  // be touching each other at t. We compute the vector between centers
  // of the balls i.e. C = b2pos - b1pos. 
  // Here C is directed from b1 to b2
  // If dot product of b1 with C is positive (b1's tangential component) or dot product of b2 with C 
  // is negative, then we can say that balls are approaching each other. 
  // Actually, if the difference of tangential components is positive
  // then the balls are approaching each other

  static bool isApproaching (const Ball& b1, const Ball& b2, const FP& t) {
    // assuming that t is a valid collision time

    Vec2 b1pos = b1.pos (t);
    Vec2 b2pos = b2.pos (t);

    Vec2 C = b2pos - b1pos;

    FP b1tanComp = C.dot (b1.vel ());
    FP b2tanComp = C.dot (b2.vel ());

    FP diff = b1tanComp - b2tanComp;

    return (diff > FP (0.0));

  }

public:
  // common code to 
  // compute earliest collision between a ball and some other object underTest
  // We don't want to create a collision with the object involved in previous collision
  template <typename I, typename T=typename std::remove_pointer<typename std::iterator_traits<I>::value_type>::type >
  static galois::optional<Event> computeNextEvent (const Event::EventKind& kind, const Ball* b, const I collObjsBeg, const I collObjsEnd, const FP& endtime, const Event* prevEvent, const Sector* sector) {

    // std::cout << "Computing future events for ball: " << b->str () << std::endl;


    galois::optional<Event> retVal;

    // const T* currMin = nullptr;
    // FP currMinTime = -1.0;

    for (I i = collObjsBeg; i != collObjsEnd; ++i) {

      const T* underTest = *i;

      // the object under test is not the same as the one involved in a previous collision event
      // if (static_cast<const CollidingObject*> (underTest) !=  static_cast<const CollidingObject*> (b)
          // && prevEventObj != underTest) { 

      if (static_cast<const CollidingObject*> (underTest) !=  static_cast<const CollidingObject*> (b)) {

        std::pair <bool, FP> p = Collision::computeCollisionTime (*b, *underTest);

        if (p.first && p.second <= endtime) {

          Event e = Event::makeEvent (kind, b, underTest, p.second, sector);


          if (prevEvent && *prevEvent == e) {
            continue;
          }

          // std::cout << "Possible future Event: " << e.str () << std::endl;

          if (retVal && FPutils::almostEqual (retVal->getTime (), e.getTime ())) {
            if (e.getOtherObj ()->getID () < retVal->getOtherObj ()->getID ()) {
              retVal = e;
            }

          } else  if (!retVal || e < *retVal) {
            retVal = e;

          } else {
            assert (retVal);
            assert (*retVal < e);
          }
        } // end if p.first
      }
    }
// 
// 
        // if (p.first) { // collision possible
// 
          // assert (p.second > FP (0.0));
// 
          // // it may happen that a ball collides two balls or
          // // two cushions simulatneously. In such cases,
          // // we break the tie by choosing the object with smaller id
          // if (FPutils::almostEqual (p.second, currMinTime)) {
            // if (underTest->getID () < currMin->getID ()) {
              // currMin = underTest;
              // currMinTime = p.second;
            // }
// 
          // } else  if ((currMin == NULL) || (p.second < currMinTime)) {
            // // colliding == NULL for the first time
            // currMin = underTest;
            // currMinTime = p.second;
// 
          // } else {
            // assert (p.second > currMinTime);
            // // do nothing?
          // }
// 
          // if (false) {
            // std::cout.precision (10);
            // std::cout << "At time: " << std::fixed << double (p.second) << " Ball b=" << b->str () << 
              // " can collide with=" << underTest->str () << std::endl;
          // }
// 
        // }
// 
// 

      // } // end outer if
    // } // end for

// 
    // if (currMin != NULL) { assert (currMinTime > FP (0.0)); }
// 
// 
    // if (currMinTime > FP (0.0) && currMinTime <= endtime) { 
      // assert (currMin != nullptr);
// 
      // retVal = Event::makeEvent (kind, b, currMin, currMinTime, sector);
// 
    // }

    return retVal;
  }


};



#endif // _COLLISION_H_
