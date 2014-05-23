#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <sstream>


#include "Galois/Galois.h"
#include "Galois/GaloisUnsafe.h"
#include "Galois/Atomic.h"
#include "Galois/Statistic.h"
#include "Galois/Runtime/DoAllCoupled.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/DAG.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;
static cll::opt<unsigned> length("len", cll::desc("Length of the array"), cll::init(10000));

const char* name = "merge sort";
const char* desc = "merge sort";
const char* url = "mergesort";

enum Algo {
  SERIAL, STL, GALOIS, CILK
};

cll::opt<Algo> algorithm (
    cll::desc ("algorithm"),
    cll::values (
      clEnumVal (SERIAL, "serial recursive"),
      clEnumVal (STL, "STL implementation"),
      clEnumVal (GALOIS, "galois divide and conquer implementation"),
      clEnumVal (CILK, "CILK divide and conquer implementation"),
      clEnumValEnd),

    cll::init (SERIAL));


static const size_t LEAF_SIZE = 8;

void initializeIntArray (int* array, const size_t L) {
  for (size_t i = 0; i < L; ++i) {
    array[i] = rand ();
  }
}

template <typename T, typename C>
void mergeHalves (T* array, T* tmp_array
    , const size_t beg, const size_t mid, const size_t end, const C& cmp ) {

  std::merge (array + beg, array + mid, array + mid, array + end, tmp_array + beg);
  std::copy (tmp_array + beg, tmp_array + end, array + beg);
}

template <typename T, typename C>
#ifdef HAVE_CILK
cilk
#endif
void splitRecursive (T* array, T* tmp_array, const size_t beg, const size_t end, const C& cmp) {
  if ((end - beg) > LEAF_SIZE) {
    const size_t mid = (beg + end) / 2;
#ifdef HAVE_CILK 
    spawn
#endif
    splitRecursive (array, tmp_array, beg, mid, cmp);
#ifdef HAVE_CILK
    spawn
#endif
    splitRecursive (array, tmp_array, mid, end, cmp);

#ifdef HAVE_CILK
    sync;
#endif

    mergeHalves (array, tmp_array, beg, mid, end, cmp);

  } else {
    std::sort (array + beg, array + end, cmp);
  }
}


template <typename T, typename C>
void mergeSortSequential (T* array, const size_t L, const C& cmp) {
  assert (L > 0);
  T* tmp_array = new T[L];
  std::copy (array + 0, array + L, tmp_array);
  splitRecursive (array, tmp_array, 0, L, cmp);
}

template <typename T, typename C>
void mergeSortCilk (T* array, const size_t L, const C& cmp) {
#ifdef HAVE_CILK
  mergeSortSequential (array, L, cmp);
#else
  std::perror ("CILK not found");
  std::abort ();
#endif
}

typedef std::pair<size_t, size_t> IndexRange;

template <typename T, typename C>
struct SplitGalois {
  
  T* array;
  T* tmp_array;
  const C& cmp;

  template <typename W>
  void operator () (const IndexRange& r, W& wl) {
    // std::printf ("running split: (%d,%d)\n", r.first, r.second);
    if ((r.second - r.first) > LEAF_SIZE) {

      const size_t mid = (r.first + r.second) / 2;
      wl.push (IndexRange (r.first, mid));
      wl.push (IndexRange (mid, r.second));

      // std::printf ("spawning split: (%d,%d)\n", r.first, mid);
      // std::printf ("spawning split: (%d,%d)\n", mid, r.second);

    } else {
      std::sort (array + r.first, array + r.second, cmp);
    }
  }

};

template <typename T, typename C>
struct MergeGalois {

  T* array;
  T* tmp_array;
  const C& cmp;

  void operator () (const IndexRange& r) {
    // std::printf ("running merge: (%d,%d)\n", r.first, r.second);
    const size_t mid = (r.first + r.second) / 2;
    mergeHalves (array, tmp_array, r.first, mid, r.second, cmp);
  }

  template <typename I>
  void operator () (const IndexRange& r, I cbeg, I cend) {

    assert (std::distance (cbeg, cend) == 2);
    const IndexRange& left = *cbeg;
    ++cbeg;
    const IndexRange& right = *cbeg;

    assert (left.first == r.first);
    assert (right.second == r.second);
    assert (left.second == right.first);

    const size_t mid = left.second;
    mergeHalves (array, tmp_array, r.first, mid, r.second, cmp);
  }

};

template <typename T, typename C>
void mergeSortGalois (T* array, const size_t L, const C& cmp) {

  T* tmp_array = new T[L];
  std::copy (array + 0, array + L, tmp_array);

  Galois::Runtime::for_each_ordered_tree_2p (
      IndexRange (0, L),
      SplitGalois<T,C> {array, tmp_array, cmp},
      MergeGalois<T,C> {array, tmp_array, cmp},
      "merge-sort-galois");

}

template <typename T, typename C>
void checkSorting (T* array, const size_t L, const C& cmp) {

  bool sorted = true;
  for (unsigned i = 0; i < (L-1); ++i) {
    if (cmp (array[i + 1], array[i])) {
      sorted = false;
      std::printf ("unordered pair: %d %d\n", array[i], array[i+1]);
      break;
    }
  }

  if (!sorted) {
    std::abort ();
  }

  std::printf ("OK, array sorted!!!\n");
}

int main (int argc, char* argv[]) {

  Galois::StatManager sm;
  LonestarStart (argc, argv, name, desc, url);
  srand (0);

  assert (length > 0);
  int* array = new int [length];

  initializeIntArray (array, length);

  Galois::StatTimer t;

  t.start ();
  switch (algorithm) {
    case SERIAL:
      mergeSortSequential (array, length, std::less<int> ());
      break;

    case STL:
      std::sort (array, array + length, std::less<int> ());
      break;

    case CILK:
      mergeSortCilk (array, length, std::less<int> ());
      break;

    case GALOIS:
      mergeSortGalois (array, length, std::less<int> ());
      break;

    default:
      std::abort ();

  }
  t.stop ();

  checkSorting (array, length, std::less<int> ());

};
