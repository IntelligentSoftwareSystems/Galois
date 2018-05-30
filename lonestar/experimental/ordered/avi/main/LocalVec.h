/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
