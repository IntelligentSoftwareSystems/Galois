/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GLOBALVEC_H_
#define GLOBALVEC_H_

#include "AuxDefs.h"

#include <vector>

#include <cstdio>
#include <cmath>

struct GlobalVec {

  //! Global vectors computed for each mesh node
  //! Q is displacement
  //! V is velocity, V_b is half step value
  //! T is time
  VecDouble vecQ;
  VecDouble vecV;
  VecDouble vecV_b;
  VecDouble vecT;
  VecDouble vecLUpdate;

  //! @param totalNDOF is total number of Mesh nodes times the dimensionality
  GlobalVec(unsigned int totalNDOF) {
    vecQ = VecDouble(totalNDOF, 0.0);

    vecV       = VecDouble(vecQ);
    vecV_b     = VecDouble(vecQ);
    vecT       = VecDouble(vecQ);
    vecLUpdate = VecDouble(vecQ);
  }

private:
  static bool computeDiff(const VecDouble& vecA, const char* nameA,
                          const VecDouble& vecB, const char* nameB,
                          bool printDiff) {
    bool result = false;
    if (vecA.size() != vecB.size()) {
      if (printDiff) {
        fprintf(
            stderr,
            "Arrays of different length %s.size () = %zd, %s.size () = %zd\n",
            nameA, vecA.size(), nameB, vecB.size());
      }
      result = false;
    } else {
      result = true; // start optimistically :)
      for (size_t i = 0; i < vecA.size(); ++i) {
        double diff = fabs(vecA[i] - vecB[i]);
        if (diff > TOLERANCE) {
          result = false;
          if (printDiff) {
            fprintf(stderr, "(%s[%zd] = %g) != (%s[%zd] = %g), diff=%g\n",
                    nameA, i, vecA[i], nameB, i, vecB[i], diff);
          } else {
            break; // no use continuing on if not printing diff;
          }
        }
      }
    }

    return result;
  }

  bool computeDiffInternal(const GlobalVec& that, bool printDiff) const {
    return true &&
           computeDiff(this->vecQ, "this->vecQ", that.vecQ, "that.vecQ",
                       printDiff) &&
           computeDiff(this->vecV, "this->vecV", that.vecV, "that.vecV",
                       printDiff) &&
           computeDiff(this->vecV_b, "this->vecV_b", that.vecV_b, "that.vecV_b",
                       printDiff) &&
           computeDiff(this->vecT, "this->vecT", that.vecT, "that.vecT",
                       printDiff) &&
           computeDiff(this->vecLUpdate, "this->vecLUpdate", that.vecLUpdate,
                       "that.vecLUpdate", printDiff);
  }

public:
  /**
   * compare the values of global vectors element by element
   *
   * @param that
   */
  bool cmpState(const GlobalVec& that) const {
    return computeDiffInternal(that, false);
  }

  /** compare the values of global vector element by element
   * and print the differences
   *
   * @param that
   */
  void printDiff(const GlobalVec& that) const {
    computeDiffInternal(that, true);
  }
};

#endif /* GLOBALVEC_H_ */
