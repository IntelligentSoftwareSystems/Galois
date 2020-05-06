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

/// \file
/// \brief Defines class DecMAP, which constructs a MAP state by decimation

#ifndef __defined_libdai_decmap_h
#define __defined_libdai_decmap_h

#include <dai/daialg.h>

namespace dai {

/// Approximate inference algorithm DecMAP, which constructs a MAP state by
/// decimation
/** Decimation involves repeating the following two steps until no free
 * variables remain:
 *  - run an approximate inference algorithm,
 *  - clamp the factor with the lowest entropy to its most probable state
 */
class DecMAP : public DAIAlgFG {
private:
  /// Stores the final MAP state
  std::vector<size_t> _state;
  /// Stores the log probability of the MAP state
  Real _logp;
  /// Maximum difference encountered so far
  Real _maxdiff;
  /// Number of iterations needed
  size_t _iters;

public:
  /// Parameters for DecMAP
  struct Properties {
    /// Verbosity (amount of output sent to stderr)
    size_t verbose;

    /// Complete or partial reinitialization of clamped subgraphs?
    bool reinit;

    /// Name of the algorithm used to calculate the beliefs on clamped subgraphs
    std::string ianame;

    /// Parameters for the algorithm used to calculate the beliefs on clamped
    /// subgraphs
    PropertySet iaopts;
  } props;

public:
  /// Default constructor
  DecMAP() : DAIAlgFG(), _state(), _logp(), _maxdiff(), _iters(), props() {}

  /// Construct from FactorGraph \a fg and PropertySet \a opts
  /** \param fg Factor graph.
   *  \param opts Parameters @see Properties
   */
  DecMAP(const FactorGraph& fg, const PropertySet& opts);

  /// \name General InfAlg interface
  //@{
  virtual DecMAP* clone() const { return new DecMAP(*this); }
  virtual DecMAP* construct(const FactorGraph& fg,
                            const PropertySet& opts) const {
    return new DecMAP(fg, opts);
  }
  virtual std::string name() const { return "DECMAP"; }
  virtual Factor belief(const Var& v) const { return beliefV(findVar(v)); }
  virtual Factor belief(const VarSet& /*vs*/) const;
  virtual Factor beliefV(size_t i) const;
  virtual Factor beliefF(size_t I) const { return belief(factor(I).vars()); }
  virtual std::vector<Factor> beliefs() const;
  virtual Real logZ() const { return _logp; }
  virtual std::vector<size_t> findMaximum() const { return _state; }
  virtual void init() {
    _maxdiff = 0.0;
    _iters   = 0;
  }
  virtual void init(const VarSet& /*ns*/) { init(); }
  virtual Real run();
  virtual Real maxDiff() const { return _maxdiff; }
  virtual size_t Iterations() const { return _iters; }
  virtual void setProperties(const PropertySet& opts);
  virtual PropertySet getProperties() const;
  virtual std::string printProperties() const;
  //@}
};

} // end of namespace dai

#endif
