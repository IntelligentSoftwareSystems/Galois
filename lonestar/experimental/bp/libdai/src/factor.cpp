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

/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found
 * in the LICENSE file.
 */

#include <dai/factor.h>

namespace dai {

using namespace std;

Factor createFactorIsing(const Var& n, Real h) {
  DAI_ASSERT(n.states() == 2);
  Real buf[2];
  buf[0] = std::exp(-h);
  buf[1] = std::exp(h);
  return Factor(n, &buf[0]);
}

Factor createFactorIsing(const Var& n1, const Var& n2, Real J) {
  DAI_ASSERT(n1.states() == 2);
  DAI_ASSERT(n2.states() == 2);
  DAI_ASSERT(n1 != n2);
  Real buf[4];
  buf[0] = (buf[3] = std::exp(J));
  buf[1] = (buf[2] = std::exp(-J));
  return Factor(VarSet(n1, n2), &buf[0]);
}

Factor createFactorExpGauss(const VarSet& ns, Real beta) {
  Factor fac(ns);
  for (size_t t = 0; t < fac.nrStates(); t++)
    fac.set(t, std::exp(rnd_stdnormal() * beta));
  return fac;
}

Factor createFactorPotts(const Var& n1, const Var& n2, Real J) {
  Factor fac(VarSet(n1, n2), 1.0);
  DAI_ASSERT(n1.states() == n2.states());
  for (size_t s = 0; s < n1.states(); s++)
    fac.set(s * (n1.states() + 1), std::exp(J));
  return fac;
}

Factor createFactorDelta(const Var& v, size_t state) {
  Factor fac(v, 0.0);
  DAI_ASSERT(state < v.states());
  fac.set(state, 1.0);
  return fac;
}

Factor createFactorDelta(const VarSet& vs, size_t state) {
  Factor fac(vs, 0.0);
  DAI_ASSERT(state < vs.nrStates());
  fac.set(state, 1.0);
  return fac;
}

} // end of namespace dai
