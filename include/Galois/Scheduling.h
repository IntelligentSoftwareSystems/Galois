// Scheduling and priority specification -*- C++ -*-

namespace Galois {
namespace Scheduling {

struct RandomOrder {
};

struct FIFO {
};

struct LIFO {
};

template<int N = 32>
struct ChunkedFIFO_N {
};

typedef ChunkedFIFO_N<32> ChunkedFIFO;

struct PushToGlobal {
};


template<typename ThisTy, typename NextTy, bool Stealing = false>
struct LocalRule {
  typedef NextTy nextRule;
  typedef ThisTy thisRule;
  enum { STEALING = Stealing };
};

template<typename ThisTy, typename NextTy>
struct ThenRule {
  typedef NextTy nextRule;
  typedef ThisTy thisRule;

  template<typename T>
  static LocalRule<T, ThenRule<ThisTy, NextTy> > thenLocally() {
    return LocalRule<T, ThenRule<ThisTy, NextTy> >();
  }

  template<typename T>
  static ThenRule<T, ThenRule<ThisTy, NextTy> > then() {
    return ThenRule<T, ThenRule<ThisTy, NextTy> >();
  }
};

template<typename ThisTy>
struct HeadRule {
  typedef ThisTy thisRule;

  template<typename T>
  static ThenRule<T, HeadRule<ThisTy> > then() {
    return ThenRule<T, HeadRule<ThisTy> >();
  }

  template<typename T>
  static LocalRule<T, HeadRule<ThisTy> > thenLocally() {
    return LocalRule<T, HeadRule<ThisTy> >();
  }
};

struct PriorityFactory {
  template<typename T>
  static HeadRule<T> first() { return HeadRule<T>(); }

  template<typename T>
  static LocalRule<PushToGlobal, T> defaultLocalRule() {
    return LocalRule<PushToGlobal, T>();
  }

  static LocalRule<PushToGlobal, HeadRule<ChunkedFIFO> > defaultOrder () {
    return defaultLocalRule<HeadRule<ChunkedFIFO> > ();
  }
};

static PriorityFactory Priority __attribute__((unused)) ;

}
}
