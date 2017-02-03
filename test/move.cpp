#include "Galois/Runtime/Bag.h"
#include "Galois/gdeque.h"
#include "Galois/gslist.h"
#include "Galois/FlatMap.h"
#ifdef GALOIS_USE_EXP
#include "Galois/ConcurrentFlatMap.h"
#endif
#include "Galois/LargeArray.h"
#include "Galois/Runtime/Mem.h"
#include "Galois/Substrate/PerThreadStorage.h"

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
  Galois::Runtime::FixedSizeHeap heap(sizeof(typename T::block_type));

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
  //test(Galois::FixedSizeBag<MoveOnly>());
  //test(Galois::ConcurrentFixedSizeBag<MoveOnly>());
  //test(Galois::FixedSizeRing<MoveOnly>());
  test(Galois::gdeque<MoveOnly>());
  test(Galois::gslist<MoveOnly>());
  test(Galois::concurrent_gslist<MoveOnly>());
  test(Galois::InsertBag<MoveOnly>());
  test(Galois::LargeArray<MoveOnly>());
  test(Galois::Substrate::PerPackageStorage<MoveOnly>());
  test(Galois::Substrate::PerThreadStorage<MoveOnly>());
#ifdef GALOIS_USE_EXP
  test(Galois::concurrent_flat_map<int, MoveOnly>());
#endif

  testContainerA(Galois::gdeque<MoveOnly>(), MoveOnly());
  testContainerAA(Galois::gslist<MoveOnly>(), MoveOnly());
  //testContainerAA(Galois::concurrent_gslist<MoveOnly>(), MoveOnly());
  testContainerA(Galois::InsertBag<MoveOnly>(), MoveOnly());
#ifdef GALOIS_USE_EXP
  testContainerB(Galois::concurrent_flat_map<int, MoveOnly>(), std::make_pair(1, MoveOnly()));
#endif
  testContainerC(Galois::gdeque<MoveOnly>(), MoveOnly());

  return 0;
}
