/** Event  -*- C++ -*-
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
 * Event.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#include "Event.h"
#include "Table.h"

#define PRINT_DEBUG false

void Event::simulate (std::vector<Event>& addList, const Table& table, const double endtime) {

  switch (kind) {

    case BALL_COLLISION:
      simulateBallCollision (downCast<Ball> (otherObj));
      addNextEvents (addList, downCast<Ball> (otherObj), table, endtime );
      break;

    case CUSHION_COLLISION:
      simulateCushionCollision (downCast<Cushion> (otherObj));
      addNextEvents (addList, downCast<Cushion> (otherObj), table, endtime);
      break;

    default:
      abort ();

  }
}

void Event::simulateCollision () {
  switch (kind) {

    case BALL_COLLISION:
      simulateBallCollision (downCast<Ball> (otherObj));
      break;

    case CUSHION_COLLISION:
      simulateCushionCollision (downCast<Cushion> (otherObj));
      break;

    default:
      abort ();

  }
}

void Event::addNextEvents (std::vector<Event>& addList, const Table& table, const double endtime) {

  switch (kind) {

    case BALL_COLLISION:
      addNextEvents (addList, downCast<Ball> (otherObj), table, endtime );
      break;

    case CUSHION_COLLISION:
      addNextEvents (addList, downCast<Cushion> (otherObj), table, endtime);
      break;

    default:
      abort ();

  }
}




void Event::simulateBallCollision (Ball& b2) {

  assert (kind == BALL_COLLISION);

  Ball& b1 = *ball;

  if ((this->collCounterA == b1.collCounter ()) && 
      (this->collCounterB == b2.collCounter ())) {

    Collision::simulateCollision (b1, b2, time);

    b1.incrCollCounter ();
    b2.incrCollCounter ();

    this->collCounterA = b1.collCounter ();
    this->collCounterB = b2.collCounter ();
  }
}


void Event::simulateCushionCollision (Cushion& c) { 

  assert (kind == CUSHION_COLLISION);

  Ball& b = *ball;

  // cushion's collision counter should not change
  assert (this->collCounterB == c.collCounter ()); 

  if (this->collCounterA == b.collCounter ()) {
    Collision::simulateCollision (b, c, time);

    b.incrCollCounter ();
    this->collCounterA = b.collCounter ();
  }
}


void Event::addNextEvents (std::vector<Event>& addList, Ball& b2, const Table& table, const double endtime) {

  assert (kind == BALL_COLLISION);
  Ball& b1 = *ball;

  // check for staleness
  if ((this->collCounterA != b1.collCounter ()) ||
      (this->collCounterB != b2.collCounter ())) {


    
    if (this->collCounterA == b1.collCounter ()) {
      // b2 has collided with something else, while
      // b1 has not collided with anything since this event was
      // created

      assert (b2.collCounter () > this->collCounterB);

      if (PRINT_DEBUG) {
        std::cout << "Re-scheduling new events for ball: " << b1.str () << std::endl;
      }

      table.addNextEvents (addList, &b1, (Ball*) NULL, endtime);

    }

    if (this->collCounterB == b2.collCounter ()) {
      // b2 has not collided with anything yet

      assert (b1.collCounter () > this->collCounterA);

      if (PRINT_DEBUG) {
        std::cout << "Re-scheduling new events for ball: " << b2.str () << std::endl;
      }

      table.addNextEvents (addList, &b2, (Ball*) NULL, endtime);
    }


  } else {

    table.addNextEvents (addList, &b1, &b2, endtime);
    table.addNextEvents (addList, &b2, &b1, endtime);

  }
}


void Event::addNextEvents (std::vector<Event>& addList, Cushion& c, const Table& table, const double endtime) {

  assert (kind == CUSHION_COLLISION);

  Ball& b = *ball;

  // cushion's collision counter should not change
  assert (this->collCounterB == c.collCounter ()); 

  if (this->collCounterA == b.collCounter ()) {
    table.addNextEvents (addList, &b, &c, endtime);
  }

}
