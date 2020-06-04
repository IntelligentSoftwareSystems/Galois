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

#ifndef LONESTAR_MATRIXCOMPLETION_H
#define LONESTAR_MATRIXCOMPLETION_H

#include <cassert>
#include <galois/gstl.h>
#include <string>
#include "llvm/Support/CommandLine.h"

typedef float LatentValue;
typedef float EdgeType;

// Purdue, CSGD: 100; Intel: 20
// static const int LATENT_VECTOR_SIZE = 100;
static const int LATENT_VECTOR_SIZE = 20; // Purdue, CSGD: 100; Intel: 20

/**
 * Common commandline parameters to for matrix completion algorithms
 */
enum OutputType { binary, ascii };

namespace cll = llvm::cl;

/*
 * (Purdue, Neflix): 0.012, (Purdue, Yahoo Music): 0.00075, (Purdue, HugeWiki):
 * 0.001 Intel: 0.001 Bottou: 0.1
 */
static cll::opt<float> learningRate("learningRate",
                                    cll::desc("learning rate parameter [alpha] "
                                              "for Bold, Bottou, Intel and "
                                              "Purdue step size function"),
                                    cll::init(0.012));

/*
 * (Purdue, Netflix): 0.015, (Purdue, Yahoo Music): 0.01,
 * (Purdue, HugeWiki): 0.0, Intel: 0.9
 */
static cll::opt<float> decayRate("decayRate",
                                 cll::desc("decay rate parameter [beta] for "
                                           "Intel and Purdue step size "
                                           "function"),
                                 cll::init(0.015));
/*
 * (Purdue, Netflix): 0.05, (Purdue, Yahoo Music): 1.0, (Purdue, HugeWiki): 0.01
 * Intel: 0.001
 */
static cll::opt<float> lambda("lambda",
                              cll::desc("regularization parameter [lambda]"),
                              cll::init(0.05));

static cll::opt<unsigned> usersPerBlock("usersPerBlock",
                                        cll::desc("users per block"),
                                        cll::init(2048));
static cll::opt<unsigned> itemsPerBlock("itemsPerBlock",
                                        cll::desc("items per block"),
                                        cll::init(350));
static cll::opt<float>
    tolerance("tolerance", cll::desc("convergence tolerance"), cll::init(0.01));

static cll::opt<bool> useSameLatentVector("useSameLatentVector",
                                          cll::desc("initialize all nodes to "
                                                    "use same latent vector"),
                                          cll::init(false));

/*
 * Regarding algorithm termination
 */
static cll::opt<unsigned> maxUpdates("maxUpdates",
                                     cll::desc("Max number of times to update "
                                               "latent vectors (default 100)"),
                                     cll::init(100));

static cll::opt<std::string>
    outputFilename(cll::Positional, cll::desc("[output file]"), cll::init(""));
static cll::opt<std::string>
    transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<OutputType>
    outputType("output", cll::desc("Output type:"),
               cll::values(clEnumValN(OutputType::binary, "binary", "Binary"),
                           clEnumValN(OutputType::ascii, "ascii", "ASCII")),
               cll::init(OutputType::binary));

static cll::opt<unsigned int>
    updatesPerEdge("updatesPerEdge", cll::desc("number of updates per edge"),
                   cll::init(1));

static cll::opt<unsigned int>
    fixedRounds("fixedRounds", cll::desc("run for a fixed number of rounds"),
                cll::init(0));
static cll::opt<bool> useExactError("useExactError",
                                    cll::desc("use exact error for testing "
                                              "convergence"),
                                    cll::init(false));
static cll::opt<bool>
    useDetInit("useDetInit",
               cll::desc("initialize all nodes to "
                         "use deterministic values for latent vector"),
               cll::init(false));

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
T innerProduct(T* __restrict__ first1,
               T* __restrict__ GALOIS_USED_ONLY_IN_DEBUG(last1),
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
  virtual ~StepFunction() {}
};

/*
 * Generate a number [-1, 1] using node id
 * for deterministic runs
 */
static double genVal(uint32_t n) {
  return 2.0 * ((double)n / (double)RAND_MAX) - 1.0;
}

StepFunction* newStepFunction();

template <typename Graph>
size_t initializeGraphData(Graph& g);

#endif
