#ifndef GALOIS_RUNTIME_DUALLEVELITERATOR_H
#define GALOIS_RUNTIME_DUALLEVELITERATOR_H

#include <iterator>

namespace GaloisRuntime {

//! Iterator over items in containers of containers
template<typename IterTy>
class DualLevelIterator: public std::iterator<std::forward_iterator_tag, typename std::iterator_traits<typename std::iterator_traits<IterTy>::value_type::iterator>::value_type > {
  IterTy oo;
  IterTy eo;

  typedef typename std::iterator_traits<IterTy>::value_type::iterator InnerIt;

  InnerIt ii;

  void follow() {
    while (oo != eo && ii == oo->end()) { 
      ++oo;
      if (oo != eo)
	ii = oo->begin();
    }
  }

public:
  typedef typename std::iterator_traits<InnerIt>::value_type value_type;

  DualLevelIterator() { }
  DualLevelIterator(const DualLevelIterator& rhs): oo(rhs.oo), eo(rhs.eo), ii(rhs.ii) { }
  DualLevelIterator(const IterTy& b, const IterTy& e): oo(b), eo(e) {
    if (oo != eo) {
      ii = oo->begin();
      follow();
    }
  }

  DualLevelIterator& operator++() {
    ++ii;
    follow();
    return *this;
  }

  DualLevelIterator operator++(int) {
    DualLevelIterator tmp(*this);
    operator++(); 
    return tmp;
  }

  bool operator==(const DualLevelIterator& rhs) const {
    return oo == rhs.oo && (oo == eo || ii == rhs.ii);
  }

  bool operator!=(const DualLevelIterator& rhs) const {
    return !(*this == rhs);
  }

  value_type& operator*() const {
    return *ii;
  }
};

}
#endif
