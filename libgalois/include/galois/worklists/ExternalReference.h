#ifndef GALOIS_WORKLIST_EXTERNALREFERENCE_H
#define GALOIS_WORKLIST_EXTERNALREFERENCE_H

namespace galois {
namespace worklists {

template<typename Container, bool IgnorePushInitial = false>
class ExternalReference {
  Container& wl;

public:
  //! change the type the worklist holds
  template<typename _T>
  using retype = ExternalReference<typename Container::template retype<_T>>;

  //! T is the value type of the WL
  typedef typename Container::value_type value_type;

  ExternalReference(Container& _wl) :wl(_wl) {}

  //! push a value onto the queue
  void push(const value_type& val) { wl.push(val); }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) { wl.push(b,e); }

  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template<typename RangeTy>
  void push_initial(const RangeTy& r) { if (!IgnorePushInitial) wl.push_initial(r); }

  //! pop a value from the queue.
  galois::optional<value_type> pop() { return wl.pop(); }
};

}
}
#endif
