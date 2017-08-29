// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef A_RADIX_INCLUDED
#define A_RADIX_INCLUDED

#include <iostream>
#include <math.h>
#include "parallel.h"
#include "sequence.h"
#include "utils.h"
#include "transpose.h"
using namespace std;

namespace intSort {

  // Cannot be greater than 8 without changing definition of bIndexT
  //    from unsigned char to unsigned int (or unsigned short)
#define MAX_RADIX 8 
#define BUCKETS 256    // 1 << MAX_RADIX

  typedef int bucketsT[BUCKETS];

  // a type that must hold MAX_RADIX bits
  typedef unsigned char bIndexT;

  template <class E, class F>
  void radixBlock(E* A, E* B, bIndexT *Tmp, bucketsT counts, bucketsT offsets,
		  int Boffset, int n, int m, F extract) {
    for (int i = 0; i < m; i++)  counts[i] = 0;
    for (int j = 0; j < n; j++) {
      int k = Tmp[j] = extract(A[j]);
      counts[k]++;
    }
    int s = Boffset;
    for (int i = 0; i < m; i++) {
      s += counts[i];
      offsets[i] = s;
    }
    for (int j = n-1; j >= 0; j--) {
      int x =  --offsets[Tmp[j]];
      B[x] = A[j];
    }
  }

  template <class E, class F>
  void radixStepSerial(E* A, E* B, bIndexT *Tmp, bucketsT buckets,
		       int n, int m, F extract) {
    radixBlock(A, B, Tmp, buckets, buckets, 0, n, m, extract);
    for (int i=0; i < n; i++) A[i] = B[i];
    return;
  }

  // A is the input and sorted output (length = n)
  // B is temporary space for copying data (length = n)
  // Tmp is temporary space for extracting the bytes (length = n)
  // BK is an array of bucket sets, each set has BUCKETS integers
  //    it is used for temporary space for bucket counts and offsets
  // numBK is the length of BK (number of sets of buckets)
  // the first entry of BK is also used to return the offset of each bucket
  // m is the number of buckets per set (m <= BUCKETS)
  // extract is a function that extract the appropriate bits from A
  //  it must return a non-negative integer less than m
  template <class E, class F>
  void radixStep(E* A, E* B, bIndexT *Tmp, bucketsT *BK,
		 int numBK, int n, int m, bool top, F extract) {

    // need 3 bucket sets per block
    int expand = (sizeof(E)<=4) ? 64 : 32;
    int blocks = min(numBK/3,(1+n/(BUCKETS*expand)));

    if (blocks < 2) {
      radixStepSerial(A, B, Tmp, BK[0], n, m, extract);
      return;
    }
    int nn = (n+blocks-1)/blocks;
    int* cnts = (int*) BK;
    int* oA = (int*) (BK+blocks);
    int* oB = (int*) (BK+2*blocks);

    parallel_doall_1 (int, i, 0, blocks) {
      int od = i*nn;
      int nni = min(max(n-od,0),nn);
      radixBlock(A+od, B, Tmp+od, cnts + m*i, oB + m*i, od, nni, m, extract);
    } parallel_doall_end

    transpose<int>(cnts, oA).trans(blocks, m);

    int ss;
    if (top)
      ss = sequence::scan(oA, oA, blocks*m, utils::addF<int>(),0);
    else
      ss = sequence::scanSerial(oA, oA, blocks*m, utils::addF<int>(),0);
    //utils::myAssert(ss == n, "radixStep: sizes don't match");

    blockTrans<E>(B, A, oB, oA, cnts).trans(blocks, m);

    // put the offsets for each bucket in the first bucket set of BK
    for (int j = 0; j < m; j++) BK[0][j] = oA[j*blocks];
  }

  // a function to extract "bits" bits starting at bit location "offset"
  template <class E, class F>
    struct eBits {
      F _f;  int _mask;  int _offset;
      eBits(int bits, int offset, F f): _mask((1<<bits)-1), 
					_offset(offset), _f(f) {}
      int operator() (E p) {return _mask&(_f(p)>>_offset);}
    };

  // Radix sort with low order bits first
  template <class E, class F>
  void radixLoopBottomUp(E *A, E *B, bIndexT *Tmp, bucketsT *BK,
			 int numBK, int n, int bits, bool top, F f) {
      int rounds = 1+(bits-1)/MAX_RADIX;
      int rbits = 1+(bits-1)/rounds;
      int bitOffset = 0;
      while (bitOffset < bits) {
	if (bitOffset+rbits > bits) rbits = bits-bitOffset;
	radixStep(A, B, Tmp, BK, numBK, n, 1 << rbits, top,
		  eBits<E,F>(rbits,bitOffset,f));
	bitOffset += rbits;
      }
  }

  // Radix sort with high order bits first
  template <class E, class F>
  void radixLoopTopDown(E *A, E *B, bIndexT *Tmp, bucketsT *BK,
			int numBK, int n, int bits, F f) {
    if (n == 0) return;
    if (bits <= MAX_RADIX) {
      radixStep(A, B, Tmp, BK, numBK, n, 1 << bits, true, eBits<E,F>(bits,0,f));
    } else if (numBK >= BUCKETS+1) {
      radixStep(A, B, Tmp, BK, numBK, n, BUCKETS, true,
		eBits<E,F>(MAX_RADIX,bits-MAX_RADIX,f));
      int* offsets = BK[0];
      int remain = numBK - BUCKETS - 1;
      float y = remain / (float) n;
//      parallel_for (int i=0; i < BUCKETS; i++) {
      parallel_doall(int, i, 0, BUCKETS)  {
	int segOffset = offsets[i];
	int segNextOffset = (i == BUCKETS-1) ? n : offsets[i+1];
	int segLen = segNextOffset - segOffset;
	int blocksOffset = ((int) floor(segOffset * y)) + i + 1;
	int blocksNextOffset = ((int) floor(segNextOffset * y)) + i + 2;
	int blockLen = blocksNextOffset - blocksOffset;
	radixLoopTopDown(A + segOffset, B + segOffset, Tmp + segOffset, 
			 BK + blocksOffset, blockLen, segLen,
			 bits-MAX_RADIX, f);
      } parallel_doall_end
    } else {
      radixLoopBottomUp(A, B, Tmp, BK, numBK, n, bits, false, f);
    }
  }

  template <class E>
  long iSortSpace(long n) {
    long numBK = 1+n/(BUCKETS*8);
    return sizeof(E)*n + sizeof(bIndexT)*n + sizeof(bucketsT)*numBK;
  }

  // Sorts the array A, which is of length n. 
  // Function f maps each element into an integer in the range [0,m)
  // If bucketOffsets is not NULL then it should be an array of length m
  // The offset in A of each bucket i in [0,m) is placed in location i
  //   such that for i < m-1, offsets[i+1]-offsets[i] gives the number 
  //   of keys=i.   For i = m-1, n-offsets[i] is the number.
  template <class E, class F>
  void iSort(E *A, int* bucketOffsets, long n, long m, bool bottomUp, 
	     char* tmpSpace, F f) {
    int bits = utils::log2Up(m);
    long numBK = 1+n/(BUCKETS*8);

    // the temporary space is broken into 3 parts: B, Tmp and BK
    E *B = (E*) tmpSpace; 
    long Bsize =sizeof(E)*n;
    bIndexT *Tmp = (bIndexT*) (tmpSpace+Bsize); // one byte per item
    long tmpSize = sizeof(bIndexT)*n;
    bucketsT *BK = (bucketsT*) (tmpSpace+Bsize+tmpSize);
    if (bits <= MAX_RADIX) {
      radixStep(A, B, Tmp, BK, numBK, n, 1 << bits, true, eBits<E,F>(bits,0,f));
      if (bucketOffsets != NULL) {
//	parallel_for (int i=0; i < m; i++) 
	parallel_doall(int, i, 0, m) {
	  bucketOffsets[i] = BK[0][i];
        } parallel_doall_end
      }
      return;
    } else if (bottomUp)
      radixLoopBottomUp(A, B, Tmp, BK, numBK, n, bits, true, f);
    else
      radixLoopTopDown(A, B, Tmp, BK, numBK, n, bits, f);
    if (bucketOffsets != NULL) {
//      {parallel_for (int i=0; i < m; i++) bucketOffsets[i] = n;}
      {parallel_doall(int, i, 0, m) { bucketOffsets[i] = n;} parallel_doall_end }
//      {parallel_for (int i=0; i < n-1; i++) {
      {
        parallel_doall(int, i, 0, n-1)  {
	  int v = f(A[i]);
	  int vn = f(A[i+1]);
	  if (v != vn) bucketOffsets[vn] = i+1;
	} parallel_doall_end
      }
      bucketOffsets[f(A[0])] = 0;
      sequence::scanIBack(bucketOffsets, bucketOffsets, (int) m,
			  utils::minF<int>(), (int) n);
    }
  }

  template <class E, class F>
  void iSort(E *A, int* bucketOffsets, int n, long m, bool bottomUp, F f) {
    long x = iSortSpace<E>(n);
    char* s = newA(char, x);
    iSort(A, bucketOffsets, n, m, bottomUp, s, f);
    free(s);
  }

  template <class E, class F>
  void iSort(E *A, int* bucketOffsets, int n, long m, F f) {
    iSort(A, bucketOffsets, n, m, false, f);}

  // A version that uses a NULL bucketOffset
  template <class E, class Func>
  void iSort(E *A, int n, long m, Func f) { 
    iSort(A, NULL, n, m, false, f);}

  template <class E, class F>
  void iSortBottomUp(E *A, int n, long m, F f) {
    iSort(A, NULL, n, m, true, f);}
};

typedef unsigned int uint;

static void integerSort(uint *A, int n) {
  long maxV = sequence::reduce(A,n,utils::maxF<uint>());
  intSort::iSort(A, NULL, n, maxV+1,  utils::identityF<uint>());
}

static void integerSort(uint *A, int n, char* s) {
  long maxV = sequence::reduce(A,n,utils::maxF<uint>());
  intSort::iSort(A, NULL, n, maxV+1, false, s, utils::identityF<uint>());
}

template <class T>
void integerSort(pair<uint,T> *A, int n) {
  long maxV = sequence::mapReduce<uint>(A,n,utils::maxF<uint>(),
					utils::firstF<uint,T>());
  intSort::iSort(A, NULL, n, maxV+1,  utils::firstF<uint,T>());
}

template <class T>
void integerSort(pair<uint,T> *A, int n, char* s) {
  long maxV = sequence::mapReduce<uint>(A,n,utils::maxF<uint>(),
					utils::firstF<uint,T>());
  intSort::iSort(A, NULL, n, maxV+1, false, s, utils::firstF<uint,T>());
}

#endif

