#include "Cushion.h"
#include "Event.h"
#include "Collision.h"


const FP Cushion::REFLECTION_COEFF = 1.0;

void Cushion::simulate (const Event& e) {

  assert (e.notStale ());
  assert (this == e.getCushion ());
  assert (e.getKind () == Event::CUSHION_COLLISION);

  Ball* b = e.getBall ();

  Collision::simulateCollision (*b, *this, e.getTime ());

}

