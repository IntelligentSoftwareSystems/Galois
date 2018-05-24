#ifndef GALOIS_NODEREFITERATOR_H
#define GALOIS_NODEREFITERATOR_H

#include "boost/iterator/iterator_adaptor.hpp"

namespace galois {

//! Modify an iterator so that *it == it
template<typename Iterator>
struct NoDerefIterator : public boost::iterator_adaptor<
  NoDerefIterator<Iterator>, Iterator, Iterator, 
  boost::use_default, const Iterator&>
{
  NoDerefIterator(): NoDerefIterator::iterator_adaptor_() { }
  explicit NoDerefIterator(Iterator it): NoDerefIterator::iterator_adaptor_(it) { }
  const Iterator& dereference() const {
    return NoDerefIterator::iterator_adaptor_::base_reference();
  }
  Iterator& dereference() {
    return NoDerefIterator::iterator_adaptor_::base_reference();
  }
};

//! Convenience function to create {@link NoDerefIterator}.
template<typename Iterator>
NoDerefIterator<Iterator> make_no_deref_iterator(Iterator it) {
  return NoDerefIterator<Iterator>(it);
}

}

#endif
