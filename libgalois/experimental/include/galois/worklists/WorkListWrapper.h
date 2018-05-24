/**  -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 */

#ifndef GALOIS_RUNTIME_WORK_LIST_WRAPPER_H
#define GALOIS_RUNTIME_WORK_LIST_WRAPPER_H
namespace galois {
namespace worklists {


template <typename WL>
class WLsizeWrapper: public WL {

  substrate::PerThreadStorage<size_t> size_cntr;

public:

  template <typename _T>
  using retype = WLsizeWrapper<typename WL::template retype<_T> >;


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


} // end namespace worklists
} // end namespace galois

#endif // GALOIS_RUNTIME_WORK_LIST_WRAPPER_H
