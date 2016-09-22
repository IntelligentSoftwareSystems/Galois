/** Wrapper around an iterator such that *it == it -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_NODEREFITERATOR_H
#define GALOIS_RUNTIME_NODEREFITERATOR_H

#include "boost/iterator/iterator_adaptor.hpp"

namespace Galois {
namespace Runtime {

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

} // namespace Runtime
} // namespace Galois

#endif
