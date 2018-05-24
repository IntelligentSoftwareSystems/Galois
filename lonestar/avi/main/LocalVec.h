#ifndef LOCALVEC_H_
#define LOCALVEC_H_

#include "AuxDefs.h"
#include "StandardAVI.h"

#include <vector>

struct LocalVec {
  typedef StandardAVI::BCImposedType BCImposedType;


  //! initial state as read from GlobalVec using gather
  MatDouble q;
  MatDouble v;
  MatDouble vb;
  MatDouble ti;

  //! updated state computed using initial state
  MatDouble qnew;
  MatDouble vnew;
  MatDouble vbnew;
  MatDouble vbinit;
  MatDouble tnew;

  //! some temporaries so that we don't need to allocate memory in every iteration
  MatDouble forcefield;
  MatDouble funcval;
  MatDouble deltaV;


  /**
   *
   * @param nrows
   * @param ncols
   */
  LocalVec (size_t nrows=0, size_t ncols=0) {
    q = MatDouble (nrows, VecDouble (ncols, 0.0));

    v          = MatDouble (q);
    vb         = MatDouble (q);
    ti         = MatDouble (q);
    qnew       = MatDouble (q);
    vnew       = MatDouble (q);
    vbnew      = MatDouble (q);
    vbinit     = MatDouble (q);
    tnew       = MatDouble (q);


    forcefield = MatDouble (q);
    funcval    = MatDouble (q);
    deltaV     = MatDouble (q);

  }



};
#endif /* LOCALVEC_H_ */
