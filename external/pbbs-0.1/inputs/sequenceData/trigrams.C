// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2010 Guy Blelloch and the PBBS team
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

#include <iostream>
#include <fstream>
#include "parallel.h"
#include "sequence.h"
#include "dataGen.h"
#include <math.h>

#define TRIGRAM_FILE "trigrams.txt"

struct nGramTable {
  int len;

  struct tableEntry {
    char str[10];
    int len;
    char chars[27];
    float probs[27];
  };

  tableEntry S[27][27];

  int index(char c) {
    if (c == '_')
      return 26;
    else
      return (c - 'a');
  }

  nGramTable() {
    ifstream ifile(TRIGRAM_FILE);
    if (!ifile.is_open()) {
      std::cout << "nGramTable: Unable to open trigram file" << endl;
      abort();
    } else {
      int i = 0;
      while (!ifile.eof()) {
        tableEntry x;
        ifile >> x.str >> x.len;
        float probSum = 0.0;
        for (int j = 0; j < x.len; j++) {
          float prob;
          ifile >> x.chars[j] >> prob;
          probSum += prob;
          if (j == x.len - 1)
            x.probs[j] = 1.0;
          else
            x.probs[j] = probSum;
        }
        int i0 = index(x.str[0]);
        int i1 = index(x.str[1]);
        if (i0 > 26 || i1 > 26)
          abort();
        S[i0][i1] = x;
        i++;
      }
      len = i;
    }
  }

  char next(char c0, char c1, int i) {
    int j        = 0;
    tableEntry E = S[index(c0)][index(c1)];
    double x     = dataGen::hash<double>(i);
    while (x > E.probs[j])
      j++;
    return E.chars[j];
  }

  int word(int i, char* a, int maxLen) {
    a[0]  = next('_', '_', i);
    a[1]  = next('_', a[0], i + 1);
    int j = 1;
    while (a[j] != '_' && j < maxLen - 1) {
      j++;
      a[j] = next(a[j - 2], a[j - 1], i + j);
    }
    a[j] = 0;
    return j + 1;
  }

  int wordLength(int i, int maxLen) {
    char a0 = next('_', '_', i);
    char a1 = next('_', a0, i + 1);
    int j   = 1;
    while (a1 != '_' && j < maxLen - 1) {
      j++;
      char tmp = next(a0, a1, i + j);
      a0       = a1;
      a1       = tmp;
    }
    return j + 1;
  }

  char* word(int i) {
    int MAX_LEN = 100;
    char a[MAX_LEN + 1];
    int l     = word(i, a, MAX_LEN);
    char* out = newA(char, l);
    for (int j = 0; j < l; j++)
      out[j] = a[j];
    return out;
  }

  char* string(int s, int e) {
    int n   = e - s;
    char* a = newA(char, n + 1);
    int j   = 0;
    while (j < n) {
      int l        = word(j + s, a + j, n - j);
      a[j + l - 1] = ' ';
      j += l;
    }
    a[n] = 0;
    return a;
  }
};

char* trigramString(int s, int e) {
  nGramTable T = nGramTable();
  return T.string(s, e);
}

void _cilk_broken(nGramTable T, char** A, char* AA, long* L, int s, int n) {
  //  parallel_for (int i = 0; i < n; i++) {
  parallel_doall(int, i, 0, n) {
    A[i] = AA + L[i];
    T.word(100 * (i + s), A[i], 100);
  }
  parallel_doall_end
}

// allocates all words one after the other
char** trigramWords(int s, int e) {
  int n        = e - s;
  char** A     = new char*[n];
  long* L      = new long[n + 1];
  nGramTable T = nGramTable();
  //  parallel_for (int i = 0; i < n; i++)
  parallel_doall(int, i, 0, n) { L[i] = T.wordLength(100 * (i + s), 100); }
  parallel_doall_end long m =
      sequence::scan(L, L, n, utils::addF<long>(), (long)0);
  char* AA = new char[m];
  _cilk_broken(T, A, AA, L, s, n);
  free(L);
  A[n] = AA;
  return A;
}

void freeWords(char** W, int n) {
  free(W[n]);
  free(W);
}
