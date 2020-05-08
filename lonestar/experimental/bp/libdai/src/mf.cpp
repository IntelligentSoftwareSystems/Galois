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

#include <iostream>
#include <sstream>
#include <map>
#include <set>
#include <dai/mf.h>
#include <dai/util.h>

namespace dai {

using namespace std;

void MF::setProperties(const PropertySet& opts) {
  DAI_ASSERT(opts.hasKey("tol"));
  DAI_ASSERT(opts.hasKey("maxiter"));

  props.tol     = opts.getStringAs<Real>("tol");
  props.maxiter = opts.getStringAs<size_t>("maxiter");
  if (opts.hasKey("verbose"))
    props.verbose = opts.getStringAs<size_t>("verbose");
  else
    props.verbose = 0U;
  if (opts.hasKey("damping"))
    props.damping = opts.getStringAs<Real>("damping");
  else
    props.damping = 0.0;
  if (opts.hasKey("init"))
    props.init = opts.getStringAs<Properties::InitType>("init");
  else
    props.init = Properties::InitType::UNIFORM;
  if (opts.hasKey("updates"))
    props.updates = opts.getStringAs<Properties::UpdateType>("updates");
  else
    props.updates = Properties::UpdateType::NAIVE;
}

PropertySet MF::getProperties() const {
  PropertySet opts;
  opts.set("tol", props.tol);
  opts.set("maxiter", props.maxiter);
  opts.set("verbose", props.verbose);
  opts.set("damping", props.damping);
  opts.set("init", props.init);
  opts.set("updates", props.updates);
  return opts;
}

string MF::printProperties() const {
  stringstream s(stringstream::out);
  s << "[";
  s << "tol=" << props.tol << ",";
  s << "maxiter=" << props.maxiter << ",";
  s << "verbose=" << props.verbose << ",";
  s << "init=" << props.init << ",";
  s << "updates=" << props.updates << ",";
  s << "damping=" << props.damping << "]";
  return s.str();
}

void MF::construct() {
  // create beliefs
  _beliefs.clear();
  _beliefs.reserve(nrVars());
  for (size_t i = 0; i < nrVars(); ++i)
    _beliefs.push_back(Factor(var(i)));
}

void MF::init() {
  if (props.init == Properties::InitType::UNIFORM)
    for (size_t i = 0; i < nrVars(); i++)
      _beliefs[i].fill(1.0);
  else
    for (size_t i = 0; i < nrVars(); i++)
      _beliefs[i].randomize();
}

Factor MF::calcNewBelief(size_t i) {
  Factor result;
  diaforeach(const Neighbor& I, nbV(i)) {
    Factor belief_I_minus_i;
    diaforeach(const Neighbor& j, nbF(I)) // for all j in I \ i
        if (j != i) belief_I_minus_i *= _beliefs[j];
    Factor f_I = factor(I);
    if (props.updates == Properties::UpdateType::NAIVE)
      f_I.takeLog(true);
    Factor msg_I_i = (belief_I_minus_i * f_I).marginal(var(i), false);
    if (props.updates == Properties::UpdateType::NAIVE)
      result *= msg_I_i.exp();
    else
      result *= msg_I_i;
  }
  result.normalize();
  return result;
}

Real MF::run() {
  if (props.verbose >= 1)
    cerr << "Starting " << identify() << "...";

  double tic = toc();

  vector<size_t> update_seq;
  update_seq.reserve(nrVars());
  for (size_t i = 0; i < nrVars(); i++)
    update_seq.push_back(i);

  // do several passes over the network until maximum number of iterations has
  // been reached or until the maximum belief difference is smaller than
  // tolerance
  Real maxDiff = INFINITY;
  for (_iters = 0; _iters < props.maxiter && maxDiff > props.tol; _iters++) {
    random_shuffle(update_seq.begin(), update_seq.end(), rnd);

    maxDiff = -INFINITY;
    diaforeach(const size_t& i, update_seq) {
      Factor nb = calcNewBelief(i);

      if (nb.hasNaNs()) {
        cerr << name() << "::run():  ERROR: new belief of variable " << var(i)
             << " has NaNs!" << endl;
        return 1.0;
      }

      if (props.damping != 0.0)
        nb = (nb ^ (1.0 - props.damping)) * (_beliefs[i] ^ props.damping);

      maxDiff     = std::max(maxDiff, dist(nb, _beliefs[i], DISTLINF));
      _beliefs[i] = nb;
    }

    if (props.verbose >= 3)
      cerr << name() << "::run:  maxdiff " << maxDiff << " after " << _iters + 1
           << " passes" << endl;
  }

  if (maxDiff > _maxdiff)
    _maxdiff = maxDiff;

  if (props.verbose >= 1) {
    if (maxDiff > props.tol) {
      if (props.verbose == 1)
        cerr << endl;
      cerr << name() << "::run:  WARNING: not converged within "
           << props.maxiter << " passes (" << toc() - tic
           << " seconds)...final maxdiff:" << maxDiff << endl;
    } else {
      if (props.verbose >= 3)
        cerr << name() << "::run:  ";
      cerr << "converged in " << _iters << " passes (" << toc() - tic
           << " seconds)." << endl;
    }
  }

  return maxDiff;
}

Factor MF::beliefV(size_t i) const { return _beliefs[i].normalized(); }

Factor MF::belief(const VarSet& ns) const {
  if (ns.size() == 0)
    return Factor();
  else if (ns.size() == 1)
    return beliefV(findVar(*(ns.begin())));
  else {
    DAI_THROW(BELIEF_NOT_AVAILABLE);
    return Factor();
  }
}

vector<Factor> MF::beliefs() const {
  vector<Factor> result;
  for (size_t i = 0; i < nrVars(); i++)
    result.push_back(beliefV(i));
  return result;
}

Real MF::logZ() const {
  Real s = 0.0;

  for (size_t i = 0; i < nrVars(); i++)
    s -= beliefV(i).entropy();
  for (size_t I = 0; I < nrFactors(); I++) {
    Factor henk;
    diaforeach(const Neighbor& j, nbF(I)) // for all j in I
        henk *= _beliefs[j];
    henk.normalize();
    Factor piet;
    piet = factor(I).log(true);
    piet *= henk;
    s -= piet.sum();
  }

  return -s;
}

void MF::init(const VarSet& ns) {
  for (size_t i = 0; i < nrVars(); i++)
    if (ns.contains(var(i))) {
      if (props.init == Properties::InitType::UNIFORM)
        _beliefs[i].fill(1.0);
      else
        _beliefs[i].randomize();
    }
}

} // end of namespace dai
