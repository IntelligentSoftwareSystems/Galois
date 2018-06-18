/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found
 * in the LICENSE file.
 */

#include <dai/trwbp.h>

#define DAI_TRWBP_FAST 1

namespace dai {

using namespace std;

void TRWBP::setProperties(const PropertySet& opts) {
  BP::setProperties(opts);

  if (opts.hasKey("nrtrees"))
    nrtrees = opts.getStringAs<size_t>("nrtrees");
  else
    nrtrees = 0;
}

PropertySet TRWBP::getProperties() const {
  PropertySet opts = BP::getProperties();
  opts.set("nrtrees", nrtrees);
  return opts;
}

string TRWBP::printProperties() const {
  stringstream s(stringstream::out);
  string sbp = BP::printProperties();
  s << sbp.substr(0, sbp.size() - 1);
  s << ",";
  s << "nrtrees=" << nrtrees << "]";
  return s.str();
}

// This code has been copied from bp.cpp, except where comments indicate
// TRWBP-specific behaviour
Real TRWBP::logZ() const {
  Real sum = 0.0;
  for (size_t I = 0; I < nrFactors(); I++) {
    sum += (beliefF(I) * factor(I).log(true)).sum(); // TRWBP/FBP
    sum += Weight(I) * beliefF(I).entropy();         // TRWBP/FBP
  }
  for (size_t i = 0; i < nrVars(); ++i) {
    Real c_i = 0.0;
    diaforeach(const Neighbor& I, nbV(i)) c_i += Weight(I);
    if (c_i != 1.0)
      sum += (1.0 - c_i) * beliefV(i).entropy(); // TRWBP/FBP
  }
  return sum;
}

// This code has been copied from bp.cpp, except where comments indicate
// TRWBP-specific behaviour
Prob TRWBP::calcIncomingMessageProduct(size_t I, bool without_i,
                                       size_t i) const {
  Real c_I = Weight(I); // TRWBP: c_I

  Factor Fprod(factor(I));
  Prob& prod = Fprod.p();
  if (props.logdomain) {
    prod.takeLog();
    prod /= c_I; // TRWBP
  } else
    prod ^= (1.0 / c_I); // TRWBP

  // Calculate product of incoming messages and factor I
  diaforeach(const Neighbor& j, nbF(I)) if (!(without_i && (j == i))) {
    const Var& v_j = var(j);
    // prod_j will be the product of messages coming into j
    // TRWBP: corresponds to messages n_jI
    Prob prod_j(v_j.states(), props.logdomain ? 0.0 : 1.0);
    diaforeach(const Neighbor& J, nbV(j)) {
      Real c_J = Weight(J); // TRWBP
      if (J != I) {         // for all J in nb(j) \ I
        if (props.logdomain)
          prod_j += message(j, J.iter) * c_J;
        else
          prod_j *= message(j, J.iter) ^ c_J;
      } else if (c_J != 1.0) { // TRWBP: multiply by m_Ij^(c_I-1)
        if (props.logdomain)
          prod_j += message(j, J.iter) * (c_J - 1.0);
        else
          prod_j *= message(j, J.iter) ^ (c_J - 1.0);
      }
    }

    // multiply prod with prod_j
    if (!DAI_TRWBP_FAST) {
      // UNOPTIMIZED (SIMPLE TO READ, BUT SLOW) VERSION
      if (props.logdomain)
        Fprod += Factor(v_j, prod_j);
      else
        Fprod *= Factor(v_j, prod_j);
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
// TRWBP-specific behaviour
void TRWBP::calcBeliefV(size_t i, Prob& p) const {
  p = Prob(var(i).states(), props.logdomain ? 0.0 : 1.0);
  diaforeach(const Neighbor& I, nbV(i)) {
    Real c_I = Weight(I);
    if (props.logdomain)
      p += newMessage(i, I.iter) * c_I;
    else
      p *= newMessage(i, I.iter) ^ c_I;
  }
}

void TRWBP::construct() {
  BP::construct();
  _weight.resize(nrFactors(), 1.0);
  sampleWeights(nrtrees);
  if (props.verbose >= 2)
    cerr << "Weights: " << _weight << endl;
}

void TRWBP::addTreeToWeights(const RootedTree& tree) {
  for (RootedTree::const_iterator e = tree.begin(); e != tree.end(); e++) {
    VarSet ij(var(e->first), var(e->second));
    size_t I = findFactor(ij);
    _weight[I] += 1.0;
  }
}

void TRWBP::sampleWeights(size_t nrTrees) {
  if (!nrTrees)
    return;

  // initialize weights to zero
  fill(_weight.begin(), _weight.end(), 0.0);

  // construct Markov adjacency graph, with edges weighted with
  // random weights drawn from the uniform distribution on the interval [0,1]
  WeightedGraph<Real> wg;
  for (size_t i = 0; i < nrVars(); ++i) {
    const Var& v_i = var(i);
    VarSet di      = delta(i);
    for (VarSet::const_iterator j = di.begin(); j != di.end(); j++)
      if (v_i < *j)
        wg[UEdge(i, findVar(*j))] = rnd_uniform();
  }

  // now repeatedly change the random weights, find the minimal spanning tree,
  // and add it to the weights
  for (size_t nr = 0; nr < nrTrees; nr++) {
    // find minimal spanning tree
    RootedTree randTree = MinSpanningTree(wg, true);
    // add it to the weights
    addTreeToWeights(randTree);
    // resample weights of the graph
    for (WeightedGraph<Real>::iterator e = wg.begin(); e != wg.end(); e++)
      e->second = rnd_uniform();
  }

  // normalize the weights and set the single-variable weights to 1.0
  for (size_t I = 0; I < nrFactors(); I++) {
    size_t sizeI = factor(I).vars().size();
    if (sizeI == 1)
      _weight[I] = 1.0;
    else if (sizeI == 2)
      _weight[I] /= nrTrees;
    else
      DAI_THROW(NOT_IMPLEMENTED);
  }
}

} // end of namespace dai
