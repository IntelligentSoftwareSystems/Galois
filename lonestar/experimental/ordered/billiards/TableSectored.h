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

#ifndef _TABLE_SECTORED_H_
#define _TABLE_SECTORED_H_

#include "Table.h"

#include "FPutils.h"
#include "GeomUtils.h"
#include "Cushion.h"
#include "Ball.h"
#include "Collision.h"
#include "Event.h"

#include "galois/optional.h"

#include "galois/gIO.h"

#include <vector>

#include <cstdlib>
#include <cstdio>
#include <ctime>

template <typename B>
class TableSectored : public Table<B> {

public:
  using SerialTable = TableSectored<Ball>;

private:
  using Base   = Table<B>;
  using Ball_t = typename Base::Ball_t;

  unsigned sectorSize;
  unsigned xSectors;
  unsigned ySectors;

  std::vector<std::vector<Sector*>> sectors;

  TableSectored& operator=(const TableSectored& that) {
    abort();
    return *this;
  }

public:
  TableSectored(unsigned numBalls, unsigned sectorSize, unsigned xSectors,
                unsigned ySectors)
      : Base(numBalls, sectorSize, xSectors, ySectors), sectorSize(sectorSize),
        xSectors(xSectors), ySectors(ySectors) {
    init();
  }

  template <typename I>
  TableSectored(const I ballsBeg, const I ballsEnd, unsigned sectorSize,
                unsigned xSectors, unsigned ySectors)
      : Base(ballsBeg, ballsEnd, sectorSize, xSectors, ySectors),
        sectorSize(sectorSize), xSectors(xSectors), ySectors(ySectors) {
    init();
  }

  template <typename B2>
  TableSectored(const TableSectored<B2>& that)
      : Base(that), sectorSize(that.sectorSize), xSectors(that.xSectors),
        ySectors(that.ySectors) {

    // balls copied over from 'that' table have pointers to sectors in 'that'
    // table
    for (Ball_t* b : this->balls) {
      b->removeAllSectors();
    }

    init();
  }

  ~TableSectored(void) {
    assert(sectors.size() == xSectors);
    for (size_t i = 0; i < sectors.size(); ++i) {
      Base::freeVecPtr(sectors[i]);
    }
  }

  size_t getNumSectors(void) const { return sectors.size(); }

  void genInitialEvents(std::vector<Event>& initEvents, const FP& endtime) {

    initEvents.clear();

    for (Ball_t* b : Base::balls) {

      addEventsForOneBall(initEvents, b, endtime);
    }
  }

  //! @param addList
  //! @param b1, ball involved in a collision just simualtedrevious collision
  //! @param b2, other object involved in a collision just simualted
  //!
  //! The idea is to avoid generating a collision event with previous object.
  //! Because the collision physics is going to generate a new collision with
  //! the same object the ball just collided with, but we have already simulated
  //! a collision and ball should be moving in another direction and pick a
  //! different collision

  template <typename C>
  void addNextEvents(const Event& e, C& addList, const FP& endtime) const {

    switch (e.getKind()) {
    case Event::BALL_COLLISION:
      addEventsForTwoBalls(addList, e.getBall(), e.getOtherBall(), endtime, &e);
      break;

    case Event::CUSHION_COLLISION:
      addEventsForOneBall(addList, e.getBall(), endtime, &e);
      break;

    case Event::SECTOR_ENTRY:
      addEventsForOneBall(addList, e.getBall(), endtime, &e);
      break;

    case Event::SECTOR_LEAVE:
      addEventsForOneBall(addList, e.getBall(), endtime, &e);
      break;

    default:
      GALOIS_DIE("unkown event kind");
    }
  }

private:
  void init(void) {
    createSectors();
    addCushionsToSectors();
    addBallsToSectors();
  }

  void createSectors(void) {
    unsigned numSectors = xSectors * ySectors;
    assert(numSectors != 0);

    sectors.clear();
    sectors.resize(xSectors, std::vector<Sector*>(ySectors, nullptr));

    unsigned idCntr = 0;

    for (size_t i = 0; i < xSectors; ++i) {
      for (size_t j = 0; j < ySectors; ++j) {

        // bottom left corner of sector
        Vec2 bl(FP(i * sectorSize), FP(j * sectorSize));

        sectors[i][j] = new Sector(idCntr, bl, FP(sectorSize));
        idCntr++;
      }
    }

    // add adjacency information

    for (int i = 0; i < xSectors; ++i) {
      for (int j = 0; j < ySectors; ++j) {

        if ((i - 1) >= 0) {
          sectors[i][j]->addNeighbor(RectSide::LEFT, sectors[i - 1][j]);
        }

        if ((j - 1) >= 0) {
          sectors[i][j]->addNeighbor(RectSide::BOTTOM, sectors[i][j - 1]);
        }

        if ((i + 1) < xSectors) {
          sectors[i][j]->addNeighbor(RectSide::RIGHT, sectors[i + 1][j]);
        }

        if ((j + 1) < ySectors) {
          sectors[i][j]->addNeighbor(RectSide::TOP, sectors[i][j + 1]);
        }
      }
    }
  }

  void addCushionsToSectors() {

    // add cusions to sectors
    for (size_t i = 0; i < xSectors; ++i) {
      for (size_t j = 0; j < ySectors; ++j) {

        if (i == 0) {
          sectors[i][j]->addCushion(Base::getCushion(RectSide::LEFT));
        }

        if (i == (xSectors - 1)) {
          assert(xSectors > 0);

          sectors[i][j]->addCushion(Base::getCushion(RectSide::RIGHT));
        }

        if (j == 0) {
          sectors[i][j]->addCushion(Base::getCushion(RectSide::BOTTOM));
        }

        if (j == (ySectors - 1)) {
          assert(ySectors > 0);
          sectors[i][j]->addCushion(Base::getCushion(RectSide::TOP));
        }
      }
    }
  }

  void addBallsToSectors() {

    // add balls to sectors
    for (Ball_t* b : Base::balls) {

      for (size_t i = 0; i < xSectors; ++i) {
        for (size_t j = 0; j < ySectors; ++j) {

          if (sectors[i][j]->intersects(b)) {
            sectors[i][j]->addBall(b);
            b->addSector(sectors[i][j]);
          }
        }
      }
    } // end for balls
  }

  galois::optional<Event> computeEarliestEvent(Ball_t* ball, const FP& endtime,
                                               const Event* prevEvent) const {

    auto range = ball->sectorRange();

    galois::optional<Event> minEvent;
    Event::Comparator cmp;

    for (auto i = range.first; i != range.second; ++i) {
      galois::optional<Event> e =
          (*i)->computeEarliestEvent(ball, endtime, prevEvent);

      if (e) {
        if (!minEvent || cmp(*e, *minEvent)) {
          minEvent = e;
        }
      }
    }

    return minEvent;
  }

  template <typename C>
  void addEventsForTwoBalls(C& addList, Ball_t* b1, Ball_t* b2,
                            const FP& endtime, const Event* prevEvent) const {

    assert(prevEvent != nullptr);
    assert(prevEvent->getKind() == Event::BALL_COLLISION);

    if (!prevEvent->firstBallChanged()) {

      galois::optional<Event> minE1 =
          computeEarliestEvent(b1, endtime, prevEvent);
      if (minE1) {
        addList.push_back(*minE1);

        if (prevEvent->notStale()) {
          assert(*prevEvent != *minE1);
        }
      }
    }

    if (!prevEvent->otherBallChanged()) {

      galois::optional<Event> minE2 =
          computeEarliestEvent(b2, endtime, prevEvent);
      if (minE2) {
        addList.push_back(*minE2);

        if (prevEvent->notStale()) {
          assert(*prevEvent != *minE2);
        }
      }
    }
  }

  template <typename C>
  void addEventsForOneBall(C& addList, Ball_t* b, const FP& endtime,
                           const Event* prevEvent = nullptr) const {

    if (prevEvent && prevEvent->firstBallChanged()) {
      assert(prevEvent->getKind() != Event::BALL_COLLISION);
      return; // prevEvent involved a single ball and was invalid,
      // which means ball underwent some other valid event, which should have
      // scheduled a correct new event after updating the ball.
    }

    galois::optional<Event> minE = computeEarliestEvent(b, endtime, prevEvent);
    if (minE) {
      addList.push_back(*minE);

      if (prevEvent && prevEvent->notStale()) {
        assert(*prevEvent != *minE);
      }
    }
  }
};

#endif //  _TABLE_SECTORED_H_
