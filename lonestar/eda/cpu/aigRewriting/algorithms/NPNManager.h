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

/*

 @Vinicius Possani
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#ifndef NPNMANAGER_H_
#define NPNMANAGER_H_

#include <stdlib.h>
#include <string.h>
#include <assert.h>

namespace algorithm {

class NPNManager {

private:
  int nFuncs;
  char* phases;           // canonical phases
  char* perms;            // canonical permutations
  unsigned char* map;     // mapping of functions into class numbers
  unsigned short* mapInv; // mapping of classes into functions
  unsigned short* canons; // canonical forms
  char** perms4;          // four-var permutations
  char* practical;        // practical NPN classes
  static const unsigned short rewritePracticalClasses[136];

  char** getPermutations(int n);
  void getPermutationsRec(char** pRes, int nFact, int n, char Array[]);
  void truthPermuteInt(int* pMints, int nMints, char* pPerm, int nVars,
                       int* pMintsP);
  unsigned truthPermute(unsigned Truth, char* pPerms, int nVars, int fReverse);
  unsigned truthPolarize(unsigned uTruth, int Polarity, int nVars);
  void initializePractical();
  void** arrayAlloc(int nCols, int nRows, int Size);
  int factorial(int n);

public:
  NPNManager();

  ~NPNManager();

  int getNFuncs();
  unsigned short* getCanons();
  char* getPhases();
  char* getPerms();
  char* getPractical();
  unsigned char* getMap();
  unsigned short* getMapInv();
  char** getPerms4();
};

} /* namespace algorithm */

#endif /* NPNMANAGER_H_ */
