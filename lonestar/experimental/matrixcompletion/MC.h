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

#ifndef _SGD_H_
#define _SGD_H_

#include <cassert>
#include <galois/gstl.h>
#include <string>

typedef double LatentValue;

// Purdue, CSGD: 100; Intel: 20
// static const int LATENT_VECTOR_SIZE = 100;
static const int LATENT_VECTOR_SIZE = 20; // Purdue, CSGD: 100; Intel: 20

/**
 * Inner product of 2 vectors.
 *
 * Like std::inner_product but rewritten here to check vectorization
 *
 * @param first1 Pointer to beginning of vector 1
 * @param last1 Pointer to end of vector 1. Should be exactly LATENT_VECTOR_SIZE
 * away from first1
 * @param first2 Pointer to beginning of vector 2. Should have at least
 * LATENT_VECTOR_SIZE elements from this point.
 * @param init Initial value to accumulate sum into
 *
 * @returns init + the inner product (i.e. the inner product if init is 0, error
 * if init is -"ground truth"
 */
template <typename T>
T innerProduct(T* __restrict__ first1, T* __restrict__ last1,
               T* __restrict__ first2, T init) {
  assert(first1 + LATENT_VECTOR_SIZE == last1);
  for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
    init += first1[i] * first2[i];
  }
  return init;
}

template <typename T>
T predictionError(T* __restrict__ itemLatent, T* __restrict__ userLatent,
                  double actual) {
  T v = actual;
  return innerProduct(itemLatent, itemLatent + LATENT_VECTOR_SIZE, userLatent,
                      -v);
}

/**
 * Objective: squared loss with weighted-square-norm regularization
 *
 * Updates latent vectors to reduce the error from the edge value.
 *
 * @param itemLatent latent vector of the item
 * @param userLatent latent vector of the user
 * @param lambda learning parameter
 * @param edgeRating Data on the edge, i.e. the number that the inner product
 * of the 2 latent vectors should eventually get to
 * @param stepSize learning parameter: how much to adjust vectors by to
 * correct for erro
 *
 * @return Error before gradient update
 */
template <typename T>
T doGradientUpdate(T* __restrict__ itemLatent, T* __restrict__ userLatent,
                   double lambda, double edgeRating, double stepSize) {
  // Implicit cast to type T
  T l      = lambda;
  T step   = stepSize;
  T rating = edgeRating;
  T error  = innerProduct(itemLatent, itemLatent + LATENT_VECTOR_SIZE,
                         userLatent, -rating);

  // Take gradient step to reduce error
  for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
    T prevItem = itemLatent[i];
    T prevUser = userLatent[i];
    itemLatent[i] -= step * (error * prevUser + l * prevItem);
    userLatent[i] -= step * (error * prevItem + l * prevUser);
  }

  return error;
}

struct StepFunction {
  virtual LatentValue stepSize(int round) const = 0;
  virtual std::string name() const              = 0;
  virtual bool isBold() const { return false; }
};

StepFunction* newStepFunction();

template <typename Graph>
size_t initializeGraphData(Graph& g);

#endif
