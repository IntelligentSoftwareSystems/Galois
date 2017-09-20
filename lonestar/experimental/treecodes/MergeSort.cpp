#include <iostream>

#include "Galois/Galois.h"
#include "Galois/CilkInit.h"
#include "Galois/Timer.h"
#include "Galois/Runtime/TreeExec.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"


namespace cll = llvm::cl;
static cll::opt<unsigned> length("len", cll::desc("Length of the array"), cll::init(10000));
static cll::opt<unsigned> LEAF_SIZE("leaf", cll::desc("recursion leaf size"), cll::init(64));

const char* name = "merge sort";
const char* desc = "merge sort";
const char* url = "mergesort";

enum Algo {
  SERIAL, STL, CILK, GALOIS, GALOIS_STACK, GALOIS_GENERIC
};

cll::opt<Algo> algorithm (
    cll::desc ("algorithm"),
    cll::values (
      clEnumVal (SERIAL, "serial recursive"),
      clEnumVal (STL, "STL implementation"),
      clEnumVal (CILK, "CILK divide and conquer implementation"),
      clEnumVal (GALOIS, "galois divide and conquer implementation"),
      clEnumVal (GALOIS_STACK, "galois stack-based typeddivide and conquer implementation"),
      clEnumVal (GALOIS_GENERIC, "galois stack-based generic divide and conquer implementation"),
      clEnumValEnd),

    cll::init (SERIAL));


// static const size_t LEAF_SIZE = 64;

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
void splitRecursive (T* array, T* tmp_array, const size_t beg, const size_t end, const C& cmp) {
  if ((end - beg) > LEAF_SIZE) {
    const size_t mid = (beg + end) / 2;
#ifdef HAVE_CILK 
    cilk_spawn
#endif
    splitRecursive (array, tmp_array, beg, mid, cmp);
#ifdef HAVE_CILK
    cilk_spawn
#endif
    splitRecursive (array, tmp_array, mid, end, cmp);

#ifdef HAVE_CILK
    cilk_sync;
#endif

    mergeHalves (array, tmp_array, beg, mid, end, cmp);

  } else {
    std::sort (array + beg, array + end, cmp);
  }
}


template <typename T, typename C>
void mergeSortSequential (T* array, T* tmp_array, const size_t L, const C& cmp) {
  assert (L > 0);
  splitRecursive (array, tmp_array, 0, L, cmp);
}

template <typename T, typename C>
void mergeSortCilk (T* array, T* tmp_array, const size_t L, const C& cmp) {
#ifdef HAVE_CILK
  mergeSortSequential (array, tmp_array, L, cmp);
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

#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1310
  SplitGalois (
      T* array,
      T* tmp_array,
      const C& cmp)
    :
      array (array),
      tmp_array (tmp_array),
      cmp (cmp) 
  {}
#endif

  template <typename Ctx>
  void operator () (const IndexRange& r, Ctx& ctx) {
    // std::printf ("running split: (%d,%d)\n", r.first, r.second);
    if ((r.second - r.first) > LEAF_SIZE) {

      const size_t mid = (r.first + r.second) / 2;
      ctx.spawn (IndexRange (r.first, mid));
      ctx.spawn (IndexRange (mid, r.second));

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

#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1310
  MergeGalois (
      T* array,
      T* tmp_array,
      const C& cmp)
    :
      array (array),
      tmp_array (tmp_array),
      cmp (cmp) 
  {}
#endif

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
void mergeSortGalois (T* array, T* tmp_array, const size_t L, const C& cmp) {

  galois::runtime::for_each_ordered_tree (
      IndexRange (0, L),
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1310
      SplitGalois<T,C> (array, tmp_array, cmp),
      MergeGalois<T,C> (array, tmp_array, cmp),
#else
      SplitGalois<T,C> {array, tmp_array, cmp},
      MergeGalois<T,C> {array, tmp_array, cmp},
#endif
      "merge-sort-galois");


}

template <typename T, typename C>
struct MergeSortGaloisStack {
  T* array;
  T* tmp_array;
  const C& cmp;
  size_t beg;
  size_t end;

  template <typename Ctx>
  void operator () (Ctx& ctx) {
    if ((end - beg) > LEAF_SIZE) {
      size_t mid = (beg + end) / 2;

      MergeSortGaloisStack left {array, tmp_array, cmp, beg, mid};
      ctx.spawn (left);

      MergeSortGaloisStack right {array, tmp_array, cmp, mid, end};
      ctx.spawn (right);

      ctx.sync ();

      mergeHalves (array, tmp_array, beg, mid, end, cmp);
    } else {
      std::sort (array + beg, array + end, cmp);
    }
    return;
  }
};


template <typename T, typename C>
void mergeSortGaloisStack (T* array, T* tmp_array, const size_t L, const C& cmp) {
  MergeSortGaloisStack<T,C> init {array, tmp_array, cmp, 0, L};
  galois::runtime::for_each_ordered_tree (init, "mergesort-stack");
}

template <typename T, typename C>
struct MergeGaloisGeneric: public galois::runtime::TreeTaskBase {
  T* array;
  T* tmp_array;
  const C& cmp;
  size_t beg;
  size_t mid;
  size_t end;

  MergeGaloisGeneric (
      T* array,
      T* tmp_array,
      const C& cmp,
      size_t beg,
      size_t mid,
      size_t end)
    : 
      galois::runtime::TreeTaskBase (),
      array (array),
      tmp_array (tmp_array),
      cmp (cmp),
      beg (beg),
      mid (mid),
      end (end)
  {}

  virtual void operator () (galois::runtime::TreeTaskContext& ctx) {
    mergeHalves (array, tmp_array, beg, mid, end, cmp);
  }
};

template <typename T, typename C>
struct SplitGaloisGeneric: public galois::runtime::TreeTaskBase {
  T* array;
  T* tmp_array;
  const C& cmp;
  size_t beg;
  size_t end;

  SplitGaloisGeneric (
      T* array,
      T* tmp_array,
      const C& cmp,
      size_t beg,
      size_t end)
    :
      galois::runtime::TreeTaskBase (),
      array (array),
      tmp_array (tmp_array),
      cmp (cmp),
      beg (beg),
      end (end)
  {}


  virtual void operator () (galois::runtime::TreeTaskContext& ctx) {
    if ((end - beg) > LEAF_SIZE) {
      size_t mid = (beg + end) / 2;

      SplitGaloisGeneric left {array, tmp_array, cmp, beg, mid};
      ctx.spawn (left);

      SplitGaloisGeneric right {array, tmp_array, cmp, mid, end};
      ctx.spawn (right);

      ctx.sync ();

      MergeGaloisGeneric<T,C> m {array, tmp_array, cmp, beg, mid, end};
      ctx.spawn (m);

      ctx.sync ();

      // mergeHalves (array, tmp_array, beg, mid, end, cmp);

    } else {
      std::sort (array + beg, array + end, cmp);
    }
  }
};

template <typename T, typename C>
void mergeSortGaloisGeneric (T* array, T* tmp_array, const size_t L, const C& cmp) {
  SplitGaloisGeneric<T,C> init {array, tmp_array, cmp, 0, L};
  abort();
  //FIXME: galois::runtime::for_each_ordered_tree_generic (init, "mergesort-stack");
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

  galois::StatManager sm;
  LonestarStart (argc, argv, name, desc, url);
  srand (0);

  assert (length > 0);
  int* array = new int [length];

  initializeIntArray (array, length);

  galois::StatTimer t_copy ("copying time:");
  t_copy.start ();
  int* tmp_array = new int[length];
  std::copy (array + 0, array + length, tmp_array);
  t_copy.stop ();

  galois::StatTimer t;

  t.start ();
  switch (algorithm) {
    case SERIAL:
      mergeSortSequential (array, tmp_array, length, std::less<int> ());
      break;

    case STL:
      std::sort (array, array + length, std::less<int> ());
      break;

    case CILK:
      //FIXME:      galois::CilkInit();
      mergeSortCilk (array, tmp_array, length, std::less<int> ());
      break;

    case GALOIS:
      mergeSortGalois (array, tmp_array, length, std::less<int> ());
      break;

    case GALOIS_STACK:
      mergeSortGaloisStack (array, tmp_array, length, std::less<int> ());
      break;

    case GALOIS_GENERIC:
      mergeSortGaloisGeneric (array, tmp_array, length, std::less<int> ());
      break;

    default:
      std::abort ();

  }
  t.stop ();

  checkSorting (array, length, std::less<int> ());

};
