#include "Sector.h"
#include "Event.h"
#include "Collision.h"

void Sector::simulate(const Event& e) {

  assert(e.notStale());
  assert(e.getSector() == this);
  assert(e.getKind() == Event::SECTOR_ENTRY ||
         e.getKind() == Event::SECTOR_LEAVE);

  Ball_t* b = static_cast<Ball_t*>(e.getBall());

  if (e.getKind() == Event::SECTOR_ENTRY) {

    b->updateGhostPos(e.getTime());

    // assert (intersects (b));

    b->addSector(this);
    this->addBall(b);

  } else if (e.getKind() == Event::SECTOR_LEAVE) {

    b->updateGhostPos(e.getTime());

    // assert (!intersects (b));

    b->removeSector(this);
    this->removeBall(b);

  } else {
    GALOIS_DIE("unknown event kind");
  }
}

galois::optional<Event>
Sector::computeEarliestEvent(const Ball_t* ball, const FP& endtime,
                             const Event* prevEvent) const {

  // FIXME: add logic to check for new Event being the same as prevEvent

  // Minimum of:
  // 1. earliest ball event
  // 2. earliest cushion event
  // 3. earliest sector entry
  // 4. earliest sector leaving

  galois::optional<Event> minEvent;
  Event::Comparator cmp;

  galois::optional<Event> ballColl =
      Collision::computeNextEvent(Event::BALL_COLLISION, ball, balls.begin(),
                                  balls.end(), endtime, prevEvent, this);
  minEvent = ballColl;

  galois::optional<Event> cushColl = Collision::computeNextEvent(
      Event::CUSHION_COLLISION, ball, cushions.begin(), cushions.end(), endtime,
      prevEvent, this);
  if (cushColl) {
    if (!minEvent || cmp(*cushColl, *minEvent)) {
      minEvent = cushColl;
    }
  }

  galois::optional<Event> secEntry = earliestSectorEntry(ball, endtime);

  if (secEntry) {
    if (!minEvent || cmp(*secEntry, *minEvent)) {
      minEvent = secEntry;
    }
  }

  galois::optional<Event> secLeave = earliestSectorLeave(ball, endtime);

  if (secLeave) {
    if (!minEvent || cmp(*secLeave, *minEvent)) {
      minEvent = secLeave;
    }
  }

  if (prevEvent && minEvent) {
    assert(*prevEvent != *minEvent);
  }

  return minEvent;
}

galois::optional<Event> Sector::earliestSectorEntry(const Ball_t* ball,
                                                    const FP& endtime) const {

  assert(ball);

  Sector* minSector = nullptr;
  FP minTime        = -1.0;

  for (unsigned i = 0; i < sides.size(); ++i) {

    if (!neighbors[i]) {
      continue;
    }

    if (ball->hasSector(neighbors[i])) {
      // ball already added to sector by earlier event
      assert(neighbors[i]->hasBall(ball));
      continue;
    }

    std::pair<bool, FP> p =
        Collision::computeCollisionTime(*ball, sides[i], true);

    if (p.first) {

      assert(p.second >= FP(0.0));

      if (FPutils::almostEqual(p.second, minTime)) {
        assert(minSector != nullptr);

        if (neighbors[i]->getID() < minSector->getID()) {
          minSector = neighbors[i];
          minTime   = p.second;
        }

      } else if (minSector == nullptr || p.second < minTime) {
        minSector = neighbors[i];
        minTime   = p.second;

      } else {
        assert(p.second > minTime);
      }

    } // end p.first

  } // end for

  for (unsigned i = 0; i < sides.size(); ++i) {

    if (!neighbors[i]) {
      continue;
    }

    if (ball->hasSector(neighbors[i])) {
      // ball already added to sector by earlier event
      assert(neighbors[i]->hasBall(ball));
      continue;
    }

    FP brad = ball->radius();
    // to handle entry at the corners, we compute an imaginary boundary outside
    // the actual one, that is brad away. so that entry from corners can be
    // handled. e.g., a ball entring at 45 degrees from the lower left corner of
    // a sector.
    Vec2 R = brad * (sides[i].lengthVec().leftNormal().unit());

    Vec2 outerBeg = sides[i].getBegin() + R;
    Vec2 outerEnd = sides[i].getEnd() + R;

    std::pair<bool, FP> p = Collision::computeCollisionTime(
        *ball, LineSegment(outerBeg, outerEnd), true);

    if (p.first) {

      assert(p.second >= FP(0.0));

      if (FPutils::almostEqual(p.second, minTime)) {
        assert(minSector != nullptr);

        if (neighbors[i]->getID() < minSector->getID()) {
          minSector = neighbors[i];
          minTime   = p.second;
        }

      } else if (minSector == nullptr || p.second < minTime) {
        minSector = neighbors[i];
        minTime   = p.second;

      } else {
        assert(p.second > minTime);
      }

    } // end p.first

  } // end for

  galois::optional<Event> e;

  if (minSector != nullptr) {
    assert(minTime >= FP(0.0));
    assert(!ball->hasSector(minSector));
    assert(!minSector->hasBall(ball));

    if (minTime < endtime) {
      e = Event::makeEvent(Event::SECTOR_ENTRY, ball, minSector, minTime,
                           minSector);
    }
  }

  return e;
}

galois::optional<Event> Sector::earliestSectorLeave(const Ball_t* ball,
                                                    const FP& endtime) const {

  assert(ball);

  galois::optional<Event> retVal;

  if (!ball->hasSector(this)) {
    // an earlier event removed the ball from sector
    assert(!this->hasBall(ball));

    return retVal; // early Exit
  }

  FP brad = ball->radius();

  // for sector leaving, we compute an imaginary sector boundary outside
  // the actual boundary, such that the imaginary boundary is 1 ball diameter
  // away from the actual boundary.

  FP minTime = -1.0;
  for (unsigned i = 0; i < sides.size(); ++i) {

    // assuming clockwise sides, we compute 2 vectors, each of mag 2*ball_radius
    // 1 vector is along the length while other is along the left normal.

    Vec2 unitL = sides[i].lengthVec().unit();
    Vec2 v1    = FP(2) * brad * unitL;
    Vec2 v2    = FP(2) * brad * unitL.leftNormal();

    Vec2 outerBeg = sides[i].getBegin() + v2 - v1;
    Vec2 outerEnd = sides[i].getEnd() + v2 + v1;

    LineSegment outer(outerBeg, outerEnd);

    std::pair<bool, FP> p = Collision::computeCollisionTime(*ball, outer, true);

    if (p.first) {
      assert(p.second >= FP(0.0));

      if (minTime == FP(-1.0) || p.second < minTime) {
        minTime = p.second;
      }
    }
  } // end for;

  if (minTime != FP(-1.0) && minTime < endtime) {
    assert(minTime > FP(0.0));
    assert(ball->hasSector(this));
    assert(this->hasBall(ball));

    retVal = Event::makeEvent(Event::SECTOR_LEAVE, ball, this, minTime, this);
  }

  return retVal;
}
