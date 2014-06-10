#ifndef GALOIS_RUNTIME_WORK_LIST_WRAPPER_H
#define GALOIS_RUNTIME_WORK_LIST_WRAPPER_H
namespace Galois {
namespace WorkList {


template <typename WL>
class WLsizeWrapper: public WL {

  Runtime::PerThreadStorage<size_t> size_cntr;

public:

  template <typename _T>
  struct retype { 
    typedef WLsizeWrapper<typename WL::template retype<_T>::type> type; 
  };
  

  WLsizeWrapper (): WL () {
    for (unsigned i = 0; i < size_cntr.size (); ++i) {
      *(size_cntr.getRemote (i)) = 0;
    }
  }

  void push (const typename WL::value_type& v) {
    WL::push (v);
    *(size_cntr.getLocal ()) += 1;
  }

  template <typename I>
  void push (I b, I e) {
    for (I i = b; i != e; ++i) {
      push (*i);
    }
  }

  template<typename R>
  void push_initial(const R& range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  size_t size (void) const {
    size_t s = 0;
    for (unsigned i = 0; i < size_cntr.size (); ++i) {
      s += *(size_cntr.getRemote (i));
    }
    return s;
  }

  // parallel
  void reset (void) {
    *(size_cntr.getLocal ()) = 0;
  }

  // sequential
  void reset_all (void) {
    for (unsigned i = 0; i < size_cntr.size (); ++i) {
      *(size_cntr.getRemote (i)) = 0;
    }
  }


};


} // end namespace WorkList
} // end namespace Galois

#endif // GALOIS_RUNTIME_WORK_LIST_WRAPPER_H
