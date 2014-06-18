#include "Galois/Bag.h"
#include "Galois/gdeque.h"
#include "Galois/FlatMap.h"
#ifdef USE_EXP
#include "Galois/ConcurrentFlatMap.h"
#endif
#include "Galois/LargeArray.h"
#include "Galois/Runtime/PerThreadStorage.h"

struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
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
  test(Galois::InsertBag<MoveOnly>());
  test(Galois::gdeque<MoveOnly>());
  test(Galois::Runtime::PerThreadStorage<MoveOnly>());
  test(Galois::Runtime::PerPackageStorage<MoveOnly>());
#ifdef USE_EXP
  test(Galois::concurrent_flat_map<int, MoveOnly>());
#endif
  test(Galois::LargeArray<MoveOnly>());

  testContainerA(Galois::gdeque<MoveOnly>(), MoveOnly());
  testContainerA(Galois::InsertBag<MoveOnly>(), MoveOnly());
#ifdef USE_EXP
  testContainerB(Galois::concurrent_flat_map<int, MoveOnly>(), std::make_pair(1, MoveOnly()));
#endif
  testContainerC(Galois::gdeque<MoveOnly>(), MoveOnly());

  return 0;
}
