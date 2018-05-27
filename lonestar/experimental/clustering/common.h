/** Agglomerative Clustering -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 * @author M. Amber Hassaan <m.a.hassaan@utexas.edu>
 */

#ifndef CLUSTERING_COMMONG_H
#define CLUSTERING_COMMONG_H

#include "galois/gstl.h"
#include "galois/gio.h"
#include "galois/Reduction.h"

#include <cmath>


constexpr double MATH_PI = std::acos(-1);

template <typename T>
using GVector = galois::gstl::Vector<T>;

using Counter = galois::GAccumulator<size_t>;

template <typename T>
bool recursiveTreeCheck(T* rootA, T* rootB) {

  if (!rootA && !rootB) {
    return true;
  } else if ((!rootA && rootB) || (rootA && !rootB)) {
    return false;
  } else if (rootA->descendents() != rootB->descendents()) {
    return false;
  }

  assert(rootA);
  assert(rootB);
  assert(rootA->descendents() == rootB->descendents());

  T* lA = rootA->leftChild();
  T* rA = rootA->rightChild();

  T* lB = rootB->leftChild();
  T* rB = rootB->rightChild();

  bool c0 = recursiveTreeCheck(lA, lB);
  bool c1 = recursiveTreeCheck(rA, rB);


  if (recursiveTreeCheck(lA, lB) && recursiveTreeCheck(rA, rB)) { 
    return true;

  } else if (recursiveTreeCheck(lA, rB) && recursiveTreeCheck(lB, rA)) {
    return true;
  } else {
    return false;
  }

  // shouldn't reach this point
  std::abort();
  return true;
}

template <typename T>
void verifyClusterTrees(T* rootA, T* rootB) {

  if (!rootA) { GALOIS_DIE("tree A is null"); }
  if (!rootB) { GALOIS_DIE("tree B is null"); }

  if (rootA->descendents() != rootB->descendents()) {
    GALOIS_DIE("mismatch in number of desecendants, tree A has ", rootA->descendents ()
        , " tree B has ", rootB->descendents());
  }

  if (recursiveTreeCheck(rootA, rootB)) {
    std::cout << "OK...Verification successful" << std::endl;
  } else {
    GALOIS_DIE("Tree mismatch in subtrees");
  }

}

template <typename T, typename A>
void freeBinTreeSerial(T* root, A& alloc) {
  assert(root);

  T* l = root->leftChild();
  T* r = root->rightChild();

  alloc.destruct(root);
  alloc.deallocate(root, 1);

  freeBinTreeSerial(l, alloc);
  freeBinTreeSerial(r, alloc);
}

template <typename T, typename A>
void freeBinTreeParallel(T* root, A& alloc) {
  assert(root);

  galois::for_each(galois::iterate( {root} ),
      [&] (T* root, auto& ctx) {
        T* l = root->leftChild();
        T* r = root->rightChild();

        alloc.destruct(root);
        alloc.deallocate(root, 1);

        ctx.push(l);
        ctx.push(r);
      },
      galois::loopname("freeBinTreeParallel"),
      galois::no_abort());
}


#endif// CLUSTERING_COMMONG_H
