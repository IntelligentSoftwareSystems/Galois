/** Page rank application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef LONESTAR_PAGERANK_CONSTANTS_H
#define LONESTAR_PAGERANK_CONSTANTS_H

#include <iostream>

#define DEBUG 0

const char* name = "Page Rank";
const char* url  = 0;

constexpr static const float ALPHA       = 0.85;
constexpr static const float PR_INIT_VAL = 1.0;

constexpr static const float TOLERANCE   = 1.0e-5;
constexpr static const unsigned MAX_ITER = 1000;

constexpr static const unsigned PRINT_TOP = 20;

// Type definitions

typedef float PRTy;

template <typename GNode>
struct TopPair {
  float value;
  GNode id;

  TopPair(float v, GNode i) : value(v), id(i) {}

  bool operator<(const TopPair& b) const {
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};

// Helper functions

PRTy atomicAdd(std::atomic<PRTy>& v, PRTy delta) {
  PRTy old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}

#endif
