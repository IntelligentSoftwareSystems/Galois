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

#include <dai/varset.h>

namespace dai {

using namespace std;

size_t calcLinearState(const VarSet& vs, const std::map<Var, size_t>& state) {
  size_t prod = 1;
  size_t st   = 0;
  for (VarSet::const_iterator v = vs.begin(); v != vs.end(); v++) {
    std::map<Var, size_t>::const_iterator m = state.find(*v);
    if (m != state.end())
      st += prod * m->second;
    prod *= v->states();
  }
  return st;
}

std::map<Var, size_t> calcState(const VarSet& vs, size_t linearState) {
  std::map<Var, size_t> state;
  for (VarSet::const_iterator v = vs.begin(); v != vs.end(); v++) {
    state[*v] = linearState % v->states();
    linearState /= v->states();
  }
  DAI_ASSERT(linearState == 0);
  return state;
}

} // end of namespace dai
