/*

 @Vinicius Possani
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#include "NPNManager.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

namespace algorithm {

/*
 * Static attribute with precomputed data
 * the following 135 practical NPN classes of 4-variable functions were computed
 * by considering all 4-input cuts appearing in IWLS, MCNC, and ISCAS benchmarks
 *  */
const unsigned short NPNManager::rewritePracticalClasses[136] = {
    0x0000, 0x0001, 0x0003, 0x0006, 0x0007, 0x000f, 0x0016, 0x0017, 0x0018,
    0x0019, 0x001b, 0x001e, 0x001f, 0x003c, 0x003d, 0x003f, 0x0069, 0x006b,
    0x006f, 0x007e, 0x007f, 0x00ff, 0x0116, 0x0118, 0x0119, 0x011a, 0x011b,
    0x011e, 0x011f, 0x012c, 0x012d, 0x012f, 0x013c, 0x013d, 0x013e, 0x013f,
    0x0168, 0x0169, 0x016f, 0x017f, 0x0180, 0x0181, 0x0182, 0x0183, 0x0186,
    0x0189, 0x018b, 0x018f, 0x0198, 0x0199, 0x019b, 0x01a8, 0x01a9, 0x01aa,
    0x01ab, 0x01ac, 0x01ad, 0x01ae, 0x01af, 0x01bf, 0x01e9, 0x01ea, 0x01eb,
    0x01ee, 0x01ef, 0x01fe, 0x033c, 0x033d, 0x033f, 0x0356, 0x0357, 0x0358,
    0x0359, 0x035a, 0x035b, 0x035f, 0x0368, 0x0369, 0x036c, 0x036e, 0x037d,
    0x03c0, 0x03c1, 0x03c3, 0x03c7, 0x03cf, 0x03d4, 0x03d5, 0x03d7, 0x03d8,
    0x03d9, 0x03dc, 0x03dd, 0x03de, 0x03fc, 0x0660, 0x0661, 0x0666, 0x0669,
    0x066f, 0x0676, 0x067e, 0x0690, 0x0696, 0x0697, 0x069f, 0x06b1, 0x06b6,
    0x06f0, 0x06f2, 0x06f6, 0x06f9, 0x0776, 0x0778, 0x07b0, 0x07b1, 0x07b4,
    0x07bc, 0x07f0, 0x07f2, 0x07f8, 0x0ff0, 0x1683, 0x1696, 0x1698, 0x169e,
    0x16e9, 0x178e, 0x17e8, 0x18e7, 0x19e6, 0x1be4, 0x1ee1, 0x3cc3, 0x6996,
    0x0000};

/* Computes NPN canonical forms for 4-variable functions */
NPNManager::NPNManager() {

  unsigned uTruth, phase, uPerm;
  int nFuncsAux, nClasses;
  int i, k;

  nFuncs    = (1 << 16);
  nFuncsAux = nFuncs;

  canons = (unsigned short*)malloc(sizeof(unsigned short) * nFuncsAux);
  memset(canons, 0, sizeof(unsigned short) * nFuncsAux);

  phases = (char*)malloc(sizeof(char) * nFuncsAux);
  memset(phases, 0, sizeof(char) * nFuncsAux);

  perms = (char*)malloc(sizeof(char) * nFuncsAux);
  memset(perms, 0, sizeof(char) * nFuncsAux);

  map = (unsigned char*)malloc(sizeof(unsigned char) * nFuncsAux);
  memset(map, 0, sizeof(unsigned char) * nFuncsAux);

  // mapInt will filled during the processing of precomputed graphs
  mapInv = (unsigned short*)malloc(sizeof(unsigned short) * 222);
  memset(mapInv, 0, sizeof(unsigned short) * 222);

  perms4 = getPermutations(4);

  practical = (char*)malloc(sizeof(char) * nFuncsAux);
  memset(practical, 0, sizeof(char) * nFuncsAux);
  initializePractical();

  nClasses  = 1;
  nFuncsAux = (1 << 15);

  for (uTruth = 1; uTruth < (unsigned)nFuncsAux; uTruth++) {

    // skip already assigned
    if (canons[uTruth]) {
      assert(uTruth > canons[uTruth]);
      map[~uTruth & 0xFFFF] = map[uTruth] = map[canons[uTruth]];
      continue;
    }

    map[uTruth] = nClasses++;

    for (i = 0; i < 16; i++) {
      phase = truthPolarize(uTruth, i, 4);
      for (k = 0; k < 24; k++) {
        uPerm = truthPermute(phase, perms4[k], 4, 0);
        if (canons[uPerm] == 0) {
          canons[uPerm] = uTruth;
          phases[uPerm] = i;
          perms[uPerm]  = k;

          uPerm         = ~uPerm & 0xFFFF;
          canons[uPerm] = uTruth;
          phases[uPerm] = i | 16;
          perms[uPerm]  = k;
        } else {
          assert(canons[uPerm] == uTruth);
        }
      }
      phase = truthPolarize(~uTruth & 0xFFFF, i, 4);
      for (k = 0; k < 24; k++) {
        uPerm = truthPermute(phase, perms4[k], 4, 0);
        if (canons[uPerm] == 0) {
          canons[uPerm] = uTruth;
          phases[uPerm] = i;
          perms[uPerm]  = k;

          uPerm         = ~uPerm & 0xFFFF;
          canons[uPerm] = uTruth;
          phases[uPerm] = i | 16;
          perms[uPerm]  = k;
        } else {
          assert(canons[uPerm] == uTruth);
        }
      }
    }
  }

  phases[(1 << 16) - 1] = 16;
  assert(nClasses == 222);
}

NPNManager::~NPNManager() {
  free(phases);
  free(perms);
  free(map);
  free(mapInv);
  free(canons);
  free(perms4);
  free(practical);
}

char** NPNManager::getPermutations(int n) {

  char Array[50];
  char** pRes;
  int nFact, i;
  // allocate memory
  nFact = factorial(n);
  pRes  = (char**)arrayAlloc(nFact, n, sizeof(char));
  // fill in the permutations
  for (i = 0; i < n; i++) {
    Array[i] = i;
  }
  getPermutationsRec(pRes, nFact, n, Array);

  return pRes;
}

/* Fills in the array of permutations */
void NPNManager::getPermutationsRec(char** pRes, int nFact, int n,
                                    char Array[]) {

  char** pNext;
  int nFactNext;
  int iTemp, iCur, iLast, k;

  if (n == 1) {
    pRes[0][0] = Array[0];
    return;
  }

  // get the next factorial
  nFactNext = nFact / n;
  // get the last entry
  iLast = n - 1;

  for (iCur = 0; iCur < n; iCur++) {
    // swap Cur and Last
    iTemp        = Array[iCur];
    Array[iCur]  = Array[iLast];
    Array[iLast] = iTemp;

    // get the pointer to the current section
    pNext = pRes + (n - 1 - iCur) * nFactNext;

    // set the last entry
    for (k = 0; k < nFactNext; k++)
      pNext[k][iLast] = Array[iLast];

    // call recursively for this part
    getPermutationsRec(pNext, nFactNext, n - 1, Array);

    // swap them back
    iTemp        = Array[iCur];
    Array[iCur]  = Array[iLast];
    Array[iLast] = iTemp;
  }
}

/* Permutes the given vector of minterms. */
void NPNManager::truthPermuteInt(int* pMints, int nMints, char* pPerm,
                                 int nVars, int* pMintsP) {

  int m, v;
  // clean the storage for minterms
  memset(pMintsP, 0, sizeof(int) * nMints);
  // go through minterms and add the variables
  for (m = 0; m < nMints; m++)
    for (v = 0; v < nVars; v++)
      if (pMints[m] & (1 << v))
        pMintsP[m] |= (1 << pPerm[v]);
}

/* Permutes the function. */
unsigned NPNManager::truthPermute(unsigned Truth, char* pPerms, int nVars,
                                  int fReverse) {

  unsigned Result;
  int* pMints;
  int* pMintsP;
  int nMints;
  int i, m;

  assert(nVars < 6);
  nMints  = (1 << nVars);
  pMints  = (int*)malloc(sizeof(int) * nMints);
  pMintsP = (int*)malloc(sizeof(int) * nMints);
  for (i = 0; i < nMints; i++)
    pMints[i] = i;

  truthPermuteInt(pMints, nMints, pPerms, nVars, pMintsP);

  Result = 0;
  if (fReverse) {
    for (m = 0; m < nMints; m++) {
      if (Truth & (1 << pMintsP[m])) {
        Result |= (1 << m);
      }
    }
  } else {
    for (m = 0; m < nMints; m++) {
      if (Truth & (1 << m)) {
        Result |= (1 << pMintsP[m]);
      }
    }
  }

  free(pMints);
  free(pMintsP);

  return Result;
}

/* Changes the phase of the function. */
unsigned NPNManager::truthPolarize(unsigned uTruth, int Polarity, int nVars) {

  // elementary truth tables
  static unsigned Signs[5] = {
      0xAAAAAAAA, // 1010 1010 1010 1010 1010 1010 1010 1010
      0xCCCCCCCC, // 1010 1010 1010 1010 1010 1010 1010 1010
      0xF0F0F0F0, // 1111 0000 1111 0000 1111 0000 1111 0000
      0xFF00FF00, // 1111 1111 0000 0000 1111 1111 0000 0000
      0xFFFF0000  // 1111 1111 1111 1111 0000 0000 0000 0000
  };

  unsigned uCof0, uCof1;
  int Shift, v;
  assert(nVars < 6);

  for (v = 0; v < nVars; v++) {
    if (Polarity & (1 << v)) {
      uCof0 = uTruth & ~Signs[v];
      uCof1 = uTruth & Signs[v];
      Shift = (1 << v);
      uCof0 <<= Shift;
      uCof1 >>= Shift;
      uTruth = uCof0 | uCof1;
    }
  }
  return uTruth;
}

void NPNManager::initializePractical() {

  int i;
  this->practical[0] = 1;
  for (i = 1;; i++) {
    if (rewritePracticalClasses[i] == 0) {
      break;
    }
    this->practical[rewritePracticalClasses[i]] = 1;
  }
}

/* Allocated one-memory-chunk array. */
void** NPNManager::arrayAlloc(int nCols, int nRows, int Size) {

  void** pRes;
  char* pBuffer;
  int i;
  assert(nCols > 0 && nRows > 0 && Size > 0);
  pBuffer =
      (char*)malloc(sizeof(char) * (nCols * (sizeof(void*) + nRows * Size)));
  pRes    = (void**)pBuffer;
  pRes[0] = pBuffer + nCols * sizeof(void*);
  for (i = 1; i < nCols; i++) {
    pRes[i] = (void*)((char*)pRes[0] + i * nRows * Size);
  }
  return pRes;
}

int NPNManager::factorial(int n) {

  int res = 1;
  for (int i = 1; i <= n; i++) {
    res *= i;
  }
  return res;
}

int NPNManager::getNFuncs() { return this->nFuncs; }

unsigned short* NPNManager::getCanons() { return this->canons; }

char* NPNManager::getPhases() { return this->phases; }

char* NPNManager::getPerms() { return this->perms; }

char* NPNManager::getPractical() { return this->practical; }

unsigned char* NPNManager::getMap() { return this->map; }

unsigned short* NPNManager::getMapInv() { return this->mapInv; }

char** NPNManager::getPerms4() { return this->perms4; }

} /* namespace algorithm */
