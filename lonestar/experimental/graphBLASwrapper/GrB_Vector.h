#include "galois/Galois.h"
#include "galois/LargeArray.h"

#include <fstream>

// T might not necessary
template <typename T, typename I>
class GrB_Vector {
private:
  struct ItemTy {
    I idx;
    T data;
  };
  using DenseVecTy    = galois::LargeArray<T>;
  using SparseVecTy   = galois::InsertBag<ItemTy>;
  using DynamicBitset = galois::DynamicBitSet;

  DenseVecTy denseVec;
  SparseVecTy sparseVec;
  DynamicBitset dupChecker;
  // galois::GAccumulator<size_t> spAccum;
  I size;
  bool isSparse;
  bool dupCheckMode;

public:
  GrB_Vector(int n) : size(n), isSparse(true) {}
  GrB_Vector() : isSparse(true) {}

  void Initialize(I n) {
    size     = n;
    isSparse = true;
  }

  void setSize(I n) { size = n; }
  I getSize() { return size; }

  void print() {
    for (size_t i = 0; i < size; i++)
      std::cout << i << "," << denseVec[i] << "\n";
  }

  void setDupCheckMode() {
    dupCheckMode = true;
    dupChecker.resize(size);
  }

  void unsetDupCheckMode() { dupCheckMode = false; }

  void dump() {
    std::ofstream fp("label.out");
    for (size_t i = 0; i < size; i++)
      fp << i << "," << denseVec[i] << "\n";
    fp.close();
  }

  void setElement(T newItem, I idx) {
    if (isSparse) {
      // sparseVec.push_back({idx, newItem});
      if (dupCheckMode) {
        if (dupChecker.test(idx))
          return;
        dupChecker.set(idx);
      }
      ItemTy item = {idx, newItem};
      sparseVec.emplace(item);
    } else {
      denseVec[idx] = newItem;
    }
  }

  SparseVecTy& iterateSPVec() { return sparseVec; }

  SparseVecTy& getSparseVec() { return sparseVec; }

  ItemTy getSparseElement(I idx) { return sparseVec[idx]; }

  T getDenseElement(I idx) { return denseVec[idx]; }

  void trySDConvert() {
    if (isSparse) {
      isSparse = false;

      denseVec.allocateInterleaved(size);
      galois::do_all(
          galois::iterate(sparseVec),
          [&](ItemTy& item) { denseVec[item.idx] = item.data; },
          galois::steal());
    }
  }

  void clear() {
    if (isSparse)
      sparseVec.clear();
    // else denseVec.clear();
    isSparse = true;
  }
};
