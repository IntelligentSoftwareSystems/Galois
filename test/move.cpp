#include "galois/Bag.h"
#include "galois/gdeque.h"
#include "galois/gslist.h"
#include "galois/FlatMap.h"
#ifdef GALOIS_USE_EXP
#include "galois/ConcurrentFlatMap.h"
#endif
#include "galois/LargeArray.h"
#include "galois/runtime/Mem.h"
#include "galois/Substrate/PerThreadStorage.h"

struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
};

struct MoveOnlyA {
  int *x;
  MoveOnlyA() { }
  MoveOnlyA(const MoveOnlyA&) = delete;
  MoveOnly& operator=(const MoveOnlyA&) = delete;
  ~MoveOnlyA() { }
};

template<typename T>
void test(T&& x) {
  T a = std::move(x);
  T b;
  std::swap(a, b);
  a = std::move(b);
}

template<typename T, typename U>
void testContainerA(T&& x, U&& y) {
  T a = std::move(x);
  T b;
  b = std::move(a);
  b.emplace_back(std::move(y));
}

template<typename T, typename U>
void testContainerAA(T&& x, U&& y) {
  galois::runtime::FixedSizeHeap heap(sizeof(typename T::block_type));

  T a = std::move(x);
  T b;
  b = std::move(a);
  b.emplace_front(heap, std::move(y));
  b.clear(heap);
}

template<typename T, typename U>
void testContainerB(T&& x, U&& y) {
  T a = std::move(x);
  T b;
  b = std::move(a);
  b.insert(std::move(y));
}

template<typename T, typename U>
void testContainerC(T&& x, U&& y) {
  T a = std::move(x);
  T b;
  b = std::move(a);
  b.emplace(b.begin(), std::move(y));
}

int main() {
  //test(galois::FixedSizeBag<MoveOnly>());
  //test(galois::ConcurrentFixedSizeBag<MoveOnly>());
  //test(galois::FixedSizeRing<MoveOnly>());
  test(galois::gdeque<MoveOnly>());
  test(galois::gslist<MoveOnly>());
  test(galois::concurrent_gslist<MoveOnly>());
  test(galois::InsertBag<MoveOnly>());
  test(galois::LargeArray<MoveOnly>());
  test(galois::substrate::PerPackageStorage<MoveOnly>());
  test(galois::substrate::PerThreadStorage<MoveOnly>());
#ifdef GALOIS_USE_EXP
  test(galois::concurrent_flat_map<int, MoveOnly>());
#endif

  testContainerA(galois::gdeque<MoveOnly>(), MoveOnly());
  testContainerAA(galois::gslist<MoveOnly>(), MoveOnly());
  //testContainerAA(galois::concurrent_gslist<MoveOnly>(), MoveOnly());
  testContainerA(galois::InsertBag<MoveOnly>(), MoveOnly());
#ifdef GALOIS_USE_EXP
  testContainerB(galois::concurrent_flat_map<int, MoveOnly>(), std::make_pair(1, MoveOnly()));
#endif
  testContainerC(galois::gdeque<MoveOnly>(), MoveOnly());

  return 0;
}
