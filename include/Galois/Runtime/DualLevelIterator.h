#include <iterator>

namespace GaloisRuntime {

//! Iterator over items in containers of containers
template<typename Iter>
class DualLevelIterator : public std::iterator<std::forward_iterator_tag, typename std::iterator_traits<typename std::iterator_traits<Iter>::value_type::iterator>::value_type > {
  Iter I_outer;
  Iter E_outer;

  typedef typename std::iterator_traits<Iter>::value_type::iterator IITy;

  IITy I_inner;

  void follow() {
    while (I_inner == (*I_outer).end() && I_outer != E_outer) {
      ++I_outer;
      if (I_outer != E_outer)
	I_inner = (*I_outer).begin();
    }
  }

public:
  typedef typename std::iterator_traits<IITy>::value_type value_type;
  

  DualLevelIterator() :I_outer(), E_outer(), I_inner() {}
  DualLevelIterator(const DualLevelIterator& rhs) 
    :I_outer(rhs.I_outer), E_outer(rhs.E_outer), I_inner(rhs.I_inner)
  {}
  DualLevelIterator(const Iter& b, const Iter& e)
    :I_outer(b), E_outer(e)
  {
    if (I_outer != E_outer) {
      I_inner = (*I_outer).begin();
      follow();
    }
  }

  DualLevelIterator& operator++() {
    ++I_inner;
    follow();
    return *this;
  }

  DualLevelIterator operator++(int) {
    DualLevelIterator tmp(*this);
    operator++(); 
    return tmp;
  }

  bool operator==(const DualLevelIterator& rhs) const {
    return (I_outer==rhs.I_outer && I_inner == rhs.I_inner);
  }

  bool operator!=(const DualLevelIterator& rhs) const {
    return !(*this == rhs);
  }

  value_type& operator*() const {
    return *I_inner;
  }
};

}
