/*
 * GlobalVec.h
 *
 *  Created on: Jun 20, 2011
 *      Author: amber
 */

#ifndef GLOBALVEC_H_
#define GLOBALVEC_H_

#include "AuxDefs.h"

#include <vector>

#include <cstdio>
#include <cmath>

struct GlobalVec {


  VecDouble vecQ;
  VecDouble vecV;
  VecDouble vecV_b;
  VecDouble vecT;
  VecDouble vecLUpdate;

  GlobalVec (unsigned int totalNDOF) {
    vecQ = VecDouble (totalNDOF, 0.0);

    vecV       = VecDouble (vecQ);
    vecV_b     = VecDouble (vecQ);
    vecT       = VecDouble (vecQ);
    vecLUpdate = VecDouble (vecQ);
  }

private:
  static bool computeDiff (const VecDouble& vecA, const char* nameA, const VecDouble& vecB, const char* nameB, bool printDiff) {
    bool result = false;
    if (vecA.size () != vecB.size ()) {
      if (printDiff) {
        fprintf (stderr, "Arrays of different length %s.size () = %zd, %s.size () = %zd\n", nameA, vecA.size (), nameB, vecB.size ());
      }
      result = false;
    }
    else {
      result = true; // start optimistically :)
      for (size_t i = 0; i < vecA.size (); ++i) {
        double diff = fabs (vecA[i] - vecB[i]);
        if ( diff > TOLERANCE) {
          result = false;
          if (printDiff) {
            fprintf (stderr, "(%s[%zd] = %g) != (%s[%zd] = %g), diff=%g\n",
                nameA, i, vecA[i], nameB, i, vecB[i], diff);
          }
          else {
            break; // no use continuing on if not printing diff;
          }
        }
      }
    }

    return result;
  }

  bool computeDiffInternal (const GlobalVec& that, bool printDiff) const {
    return true
    && computeDiff (this->vecQ,       "this->vecQ",       that.vecQ,       "that.vecQ",       printDiff)
    && computeDiff (this->vecV,       "this->vecV",       that.vecV,       "that.vecV",       printDiff)
    && computeDiff (this->vecV_b,     "this->vecV_b",     that.vecV_b,     "that.vecV_b",     printDiff)
    && computeDiff (this->vecT,       "this->vecT",       that.vecT,       "that.vecT",       printDiff)
    && computeDiff (this->vecLUpdate, "this->vecLUpdate", that.vecLUpdate, "that.vecLUpdate", printDiff);
  }

public:
  bool cmpState (const GlobalVec& that) const {
    return computeDiffInternal (that, false);
  }

  void printDiff (const GlobalVec& that) const {
    computeDiffInternal (that, true);
  }
};

#endif /* GLOBALVEC_H_ */
