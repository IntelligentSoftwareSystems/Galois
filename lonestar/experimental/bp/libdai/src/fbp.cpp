/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found
 * in the LICENSE file.
 */

#include <dai/fbp.h>

#define DAI_FBP_FAST 1

namespace dai {

using namespace std;

// This code has been copied from bp.cpp, except where comments indicate
// FBP-specific behaviour
Real FBP::logZ() const {
  Real sum = 0.0;
  for (size_t I = 0; I < nrFactors(); I++) {
    sum += (beliefF(I) * factor(I).log(true)).sum(); // FBP
    sum += Weight(I) * beliefF(I).entropy();         // FBP
  }
  for (size_t i = 0; i < nrVars(); ++i) {
    Real c_i = 0.0;
    diaforeach(const Neighbor& I, nbV(i)) c_i += Weight(I);
    if (c_i != 1.0)
      sum += (1.0 - c_i) * beliefV(i).entropy(); // FBP
  }
  return sum;
}

// This code has been copied from bp.cpp, except where comments indicate
// FBP-specific behaviour
Prob FBP::calcIncomingMessageProduct(size_t I, bool without_i, size_t i) const {
  Real c_I = Weight(I); // FBP: c_I

  Factor Fprod(factor(I));
  Prob& prod = Fprod.p();

  if (props.logdomain) {
    prod.takeLog();
    prod /= c_I; // FBP
  } else
    prod ^= (1.0 / c_I); // FBP

  // Calculate product of incoming messages and factor I
  diaforeach(const Neighbor& j, nbF(I)) if (!(without_i && (j == i))) {
    // prod_j will be the product of messages coming into j
    // FBP: corresponds to messages n_jI
    Prob prod_j(var(j).states(), props.logdomain ? 0.0 : 1.0);
    diaforeach(const Neighbor& J,
               nbV(j)) if (J != I) { // for all J in nb(j) \ I
      if (props.logdomain)
        prod_j += message(j, J.iter);
      else
        prod_j *= message(j, J.iter);
    }
    else if (c_I != 1.0) {
      // FBP: multiply by m_Ij^(1-1/c_I)
      if (props.logdomain)
        prod_j += newMessage(j, J.iter) * (1.0 - 1.0 / c_I);
      else
        prod_j *= newMessage(j, J.iter) ^ (1.0 - 1.0 / c_I);
    }

    // multiply prod with prod_j
    if (!DAI_FBP_FAST) {
      // UNOPTIMIZED (SIMPLE TO READ, BUT SLOW) VERSION
      if (props.logdomain)
        Fprod += Factor(var(j), prod_j);
      else
        Fprod *= Factor(var(j), prod_j);
    } else {
      // OPTIMIZED VERSION
      size_t _I = j.dual;
      // ind is the precalculated IndexFor(j,I) i.e. to x_I == k corresponds x_j
      // == ind[k]
      const ind_t& ind = index(j, _I);

      for (size_t r = 0; r < prod.size(); ++r)
        if (props.logdomain)
          prod.set(r, prod[r] + prod_j[ind[r]]);
        else
          prod.set(r, prod[r] * prod_j[ind[r]]);
    }
  }
  return prod;
}

// This code has been copied from bp.cpp, except where comments indicate
// FBP-specific behaviour
void FBP::calcNewMessage(size_t i, size_t _I) {
  // calculate updated message I->i
  size_t I = nbV(i, _I);

  Real c_I = Weight(I); // FBP: c_I

  Factor Fprod(factor(I));
  Prob& prod = Fprod.p();
  prod       = calcIncomingMessageProduct(I, true, i);

  if (props.logdomain) {
    prod -= prod.max();
    prod.takeExp();
  }

  // Marginalize onto i
  Prob marg;
  if (!DAI_FBP_FAST) {
    // UNOPTIMIZED (SIMPLE TO READ, BUT SLOW) VERSION
    if (props.inference == Properties::InfType::SUMPROD)
      marg = Fprod.marginal(var(i)).p();
    else
      marg = Fprod.maxMarginal(var(i)).p();
  } else {
    // OPTIMIZED VERSION
    marg = Prob(var(i).states(), 0.0);
    // ind is the precalculated IndexFor(i,I) i.e. to x_I == k corresponds x_i
    // == ind[k]
    const ind_t ind = index(i, _I);
    if (props.inference == Properties::InfType::SUMPROD)
      for (size_t r = 0; r < prod.size(); ++r)
        marg.set(ind[r], marg[ind[r]] + prod[r]);
    else
      for (size_t r = 0; r < prod.size(); ++r)
        if (prod[r] > marg[ind[r]])
          marg.set(ind[r], prod[r]);
    marg.normalize();
  }

  // FBP
  marg ^= c_I;

  // Store result
  if (props.logdomain)
    newMessage(i, _I) = marg.log();
  else
    newMessage(i, _I) = marg;

  // Update the residual if necessary
  if (props.updates == Properties::UpdateType::SEQMAX)
    updateResidual(i, _I, dist(newMessage(i, _I), message(i, _I), DISTLINF));
}

void FBP::construct() {
  BP::construct();
  _weight.resize(nrFactors(), 1.0);
}

} // end of namespace dai
