#include "geometry.h"
#include "octTree.h"
#include "spherical.h"
#include "nbody.h"

void test() {
  TRglobal->precompute();
  vect3d v;

  innerExpansion* In1 = new innerExpansion(TRglobal, point3d(0., .2, 0.));
  In1->addTo(point3d(0., 0., 0.), 1.);
  v = In1->force(point3d(1., 0., 0.), 1.);
  v.print();
  cout << endl;

  innerExpansion* In2 = new innerExpansion(TRglobal, point3d(0., .3, 0.));
  In2->addTo(In1);
  v = In2->force(point3d(1., 0., 0.), 1.);
  v.print();
  cout << endl;

  outerExpansion* Out2 = new outerExpansion(TRglobal, point3d(1., .3, 0.));
  Out2->addTo(In2);
  v = Out2->force(point3d(1., 0., 0.), 1.);
  v.print();
  cout << endl;

  outerExpansion* Out1 = new outerExpansion(TRglobal, point3d(1., .2, 0.));
  Out1->addTo(Out2);
  v = Out1->force(point3d(1., 0., 0.), 1.);
  v.print();
  cout << endl;
}
