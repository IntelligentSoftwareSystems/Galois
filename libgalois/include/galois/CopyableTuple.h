#ifndef __GALOIS_COPYABLE_TUPLE__
#define __GALOIS_COPYABLE_TUPLE__

namespace galois {

/**
 * Struct that contains 3 elements. Used over std::tuple as std::tuple memory
 * layout isn't guaranteed.
 */
template<typename T1, typename T2, typename T3>
struct TupleOfThree {
  T1 first;
  T2 second;
  T3 third;

  // empty constructor
  TupleOfThree() { }

  // initialize 3 fields
  TupleOfThree(T1 one, T2 two, T3 three) {
    first = one;
    second = two;
    third = three;
  }
};

} // end galois namespace

#endif
