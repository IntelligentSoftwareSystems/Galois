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

#ifndef A_HASH_INCLUDED
#define A_HASH_INCLUDED

#include "parallel.h"
#include "utils.h"
#include "sequence.h"
using namespace std;

// A "history independent" hash table that supports insertion, and searching
// It is described in the paper
//   Guy E. Blelloch, Daniel Golovin
//   Strongly History-Independent Hashing with Applications
//   FOCS 2007: 272-282
// At any quiescent point (when no operations are actively updating the
//   structure) the state will depend only on the keys it contains and not
//   on the history of the insertion order.
// Insertions can happen in parallel, but they cannot overlap with searches
// Searches can happen in parallel
// Deletions must happen sequentially
template <class HASH>
class Table {
private:
  typedef typename HASH::eType eType;
  typedef typename HASH::kType kType;
  int m;
  int mask;
  eType empty;
  HASH hashStruct;
  eType* TA;
  int* compactL;

  // needs to be in separate routine due to Cilk bugs
  static void clearA(eType* A, int n, eType v) {
    //    parallel_for (int i=0; i < n; i++) A[i] = v;
    parallel_doall(int, i, 0, n) { A[i] = v; }
    parallel_doall_end
  }

  struct notEmptyF {
    eType e;
    notEmptyF(eType _e) : e(_e) {}
    int operator()(eType a) { return e != a; }
  };

  int hashToRange(unsigned int h) { return h & mask; }
  int firstIndex(kType v) { return hashToRange(hashStruct.hash(v)); }
  int incrementIndex(int h) { return hashToRange(h + 1); }
  int decrementIndex(int h) { return hashToRange(h - 1); }
  bool lessIndex(int a, int b) { return 2 * hashToRange(a - b) > m; }

public:
  // Size is the maximum number of values the hash table will hold.
  // Overfilling the table could put it into an infinite loop.
  Table(int size, HASH hashF)
      : m(1 << utils::log2Up(100 + 2 * size)), mask(m - 1),
        empty(hashF.empty()), hashStruct(hashF), TA(newA(eType, m)),
        compactL(NULL) {
    clearA(TA, m, empty);
  }

  // Deletes the allocated arrays
  void del() {
    free(TA);
    if (compactL != NULL)
      free(compactL);
  }

  // prioritized linear probing
  //   a new key will bump an existing key up if it has a higher priority
  //   an equal key will replace an old key if replaceQ(new,old) is true
  // returns 0 if not inserted (i.e. equal and replaceQ false) and 1 otherwise
  bool insert(eType v) {
    kType vkey = hashStruct.getKey(v);
    int h      = firstIndex(vkey);
    while (1) {
      eType c;
      int cmp;
      bool swapped = 0;
      c            = TA[h];
      cmp = (c == empty) ? 1 : hashStruct.cmp(vkey, hashStruct.getKey(c));

      // while v is higher priority than entry at TA[h] try to swap it in
      while (cmp == 1 && !(swapped = utils::CAS(&TA[h], c, v))) {
        c   = TA[h];
        cmp = hashStruct.cmp(vkey, hashStruct.getKey(c));
      }

      // if swap succeeded either we are done (if swapped with empty)
      // or we have a new lower priority value we have to insert
      if (swapped) {
        if (c == empty)
          return 1; // done
        else {
          v    = c;
          vkey = hashStruct.getKey(v);
        } // new value to insert

      } else {
        // if swap did not succeed then priority of TA[h] >= priority of v

        // if equal keys (priorities equal) then either quit or try to replace
        while (cmp == 0) {
          // if other equal element does not need to be replaced then quit
          if (!hashStruct.replaceQ(v, c))
            return 0;

          // otherwise try to replace (atomically) and quit if successful
          else if (utils::CAS(&TA[h], c, v))
            return 1;

          // otherwise failed due to concurrent write, try again
          c   = TA[h];
          cmp = hashStruct.cmp(vkey, hashStruct.getKey(c));
        }
      }

      // move to next bucket
      h = incrementIndex(h);
    }
    return 0; // should never get here
  }

  // needs to be more thoroughly tested
  // currently always returns true
  bool deleteVal(kType v) {
    int i = firstIndex(v);
    int cmp;

    // find first element less than or equal to v in priority order
    int j   = i;
    eType c = TA[j];
    while ((cmp = (c == empty) ? 1 : hashStruct.cmp(v, hashStruct.getKey(c))) <
           0) {
      j = incrementIndex(j);
      c = TA[j];
    }
    do {
      if (cmp > 0) {
        // value at j is less than v, need to move down one
        if (j == i)
          return true;
        j = decrementIndex(j);
      } else { // (cmp == 0)
               // found the element to delete at location j

        // Find next available element to fill location j.
        // This is a little tricky since we need to skip over elements for
        // which the hash index is greater than j, and need to account for
        // things being moved around by others as we search.
        // Makes use of the fact that values in a cell can only decrease
        // during a delete phase as elements are moved from the right to left.
        int jj  = incrementIndex(j);
        eType x = TA[jj];
        while (x != empty && lessIndex(j, firstIndex(hashStruct.getKey(x)))) {
          jj = incrementIndex(jj);
          x  = TA[jj];
        }
        int jjj = decrementIndex(jj);
        while (jjj != j) {
          eType y = TA[jjj];
          if (y == empty || !lessIndex(j, firstIndex(hashStruct.getKey(y))))
            x = y;
          jjj = decrementIndex(jjj);
        }

        // try to copy the the replacement element into j
        if (utils::CAS(&TA[j], c, x)) {
          // swap was successful
          // if the replacement element was empty, we are done
          if (x == empty)
            return true;

          // Otherwise there are now two copies of the replacement element x
          // delete one copy (probably the original) by starting to look at jj.
          // Note that others can come along in the meantime and delete
          // one or both of them, but that is fine.
          v = hashStruct.getKey(x);
          j = jj;
        } else {
          // if fails then c (with value v) has been deleted or moved to a lower
          // location by someone else.
          // start looking at one location lower
          if (j == i)
            return true;
          j = decrementIndex(j);
        }
      }
      c   = TA[j];
      cmp = (c == empty) ? 1 : hashStruct.cmp(v, hashStruct.getKey(c));
    } while (cmp >= 0);
    return true;
  }

  // Returns the value if an equal value is found in the table
  // otherwise returns the "empty" element.
  // due to prioritization, can quit early if v is greater than cell
  eType find(kType v) {
    int h   = firstIndex(v);
    eType c = TA[h];
    while (1) {
      if (c == empty)
        return empty;
      int cmp = hashStruct.cmp(v, hashStruct.getKey(c));
      if (cmp >= 0)
        if (cmp == 1)
          return empty;
        else
          return c;
      h = incrementIndex(h);
      c = TA[h];
    }
  }

  // returns the number of entries
  int count() {
    return sequence::mapReduce<int>(TA, m, utils::addF<int>(),
                                    notEmptyF(empty));
  }

  // returns all the current entries compacted into a sequence
  _seq<eType> entries() {
    bool* FL = newA(bool, m);
    //    parallel_for (int i=0; i < m; i++)
    eType e  = empty;
    eType* T = TA;
    parallel_doall(int, i, 0, m) { FL[i] = (T[i] != e); }
    parallel_doall_end _seq<eType> R = sequence::pack(TA, FL, m);
    free(FL);
    return R;
  }

  // prints the current entries along with the index they are stored at
  void print() {
    cout << "vals = ";
    for (int i = 0; i < m; i++)
      if (TA[i] != empty)
        cout << i << ":" << TA[i] << ",";
    cout << endl;
  }
};

template <class HASH, class ET>
_seq<ET> removeDuplicates(_seq<ET> S, int m, HASH hashF) {
  Table<HASH> T(m, hashF);
  ET* A = S.A;
  //  {parallel_for(int i = 0; i < S.n; i++) { T.insert(A[i]);}}
  {parallel_doall(int, i, 0, S.n){T.insert(A[i]);
}
parallel_doall_end
}
_seq<ET> R = T.entries();
T.del();
return R;
}

template <class HASH, class ET>
_seq<ET> removeDuplicates(_seq<ET> S, HASH hashF) {
  return removeDuplicates(S, S.n, hashF);
}

struct hashInt {
  typedef int eType;
  typedef int kType;
  eType empty() { return -1; }
  kType getKey(eType v) { return v; }
  unsigned int hash(kType v) { return utils::hash(v); }
  int cmp(kType v, kType b) { return (v > b) ? 1 : ((v == b) ? 0 : -1); }
  bool replaceQ(eType v, eType b) { return 0; }
};

// works for non-negative integers (uses -1 to mark cell as empty)
static _seq<int> removeDuplicates(_seq<int> A) {
  return removeDuplicates(A, hashInt());
}

typedef Table<hashInt> IntTable;
static IntTable makeIntTable(int m) { return IntTable(m, hashInt()); }

struct hashStr {
  typedef char* eType;
  typedef char* kType;

  eType empty() { return NULL; }
  kType getKey(eType v) { return v; }

  unsigned int hash(kType s) {
    unsigned int hash = 0;
    while (*s)
      hash = *s++ + (hash << 6) + (hash << 16) - hash;
    return hash;
  }

  int cmp(kType s, kType s2) {
    while (*s && *s == *s2) {
      s++;
      s2++;
    };
    return (*s > *s2) ? 1 : ((*s == *s2) ? 0 : -1);
  }

  bool replaceQ(eType s, eType s2) { return 0; }
};

static _seq<char*> removeDuplicates(_seq<char*> S) {
  return removeDuplicates(S, hashStr());
}

typedef Table<hashStr> StrTable;
static StrTable makeStrTable(int m) { return StrTable(m, hashStr()); }

template <class KEYHASH, class DTYPE>
struct hashPair {
  KEYHASH keyHash;
  typedef typename KEYHASH::kType kType;
  typedef pair<kType, DTYPE>* eType;
  eType empty() { return NULL; }

  hashPair(KEYHASH _k) : keyHash(_k) {}

  kType getKey(eType v) { return v->first; }

  unsigned int hash(kType s) { return keyHash.hash(s); }
  int cmp(kType s, kType s2) { return keyHash.cmp(s, s2); }

  bool replaceQ(eType s, eType s2) { return s->second > s2->second; }
};

static _seq<pair<char*, int>*> removeDuplicates(_seq<pair<char*, int>*> S) {
  return removeDuplicates(S, hashPair<hashStr, int>(hashStr()));
}

#endif // _A_HASH_INCLUDED
