#include "Ball.h"
#include "Event.h"
#include "Collision.h"

void Ball::simulate(const Event& e) {
  assert(e.notStale());
  assert(e.getKind() == Event::BALL_COLLISION);
  assert(this == e.getOtherBall());

  Ball* b1 = e.getBall();

  Collision::simulateCollision(*this, *b1, e.getTime());
}
