/** Markable object -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section e.g. Mark for removal, 
 *
 * Billiards Simulation Finding Partial Order
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_UTIL_MARKED_H
#define GALOIS_UTIL_MARKED_H

#include <climits>

#include <boost/iterator/counting_iterator.hpp>

#include "Galois/Galois.h"

#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/DoAllCoupled.h"

template <typename T>
struct Markable {
  private:
    static const unsigned MAX_VAL = UINT_MAX;

    T m_val;
    unsigned m_ver;

  public:

    explicit Markable (T val)
      : m_val (val), m_ver (MAX_VAL) 
    {}

    void mark (unsigned v) {
      assert (v < MAX_VAL);
      m_ver = v;
    }

    bool marked () const { return (m_ver < MAX_VAL); }

    unsigned version () const { return m_ver; }

    T& get () { return m_val; }

    const T& get () const { return m_val; }

    operator T& () { return get (); }

    operator const T& () const { return get (); }

};

template <typename T>
struct IsNotMarked {
  bool operator () (const Markable<T>& x) const {
    return !x.marked ();
  }
};


template <typename T, typename C>
struct RemoveMarked {

  typedef Galois::Runtime::PerThreadWorkList<Markable<T>, C> WL_ty;

  WL_ty& wl;

  RemoveMarked (WL_ty& _wl)
    : wl (_wl) {}

  void operator () (unsigned r) {
    assert (r < wl.numRows ());

    typename WL_ty::local_iterator new_end =
      std::partition (wl[r].begin (), wl[r].end (), IsNotMarked<T> ());

    wl[r].erase (new_end, wl[r].end ());

  }

};

template <typename T, typename C>
void removeMarked (Galois::Runtime::PerThreadWorkList<Markable<T>, C>& wl) {

  Galois::do_all (
  // Galois::Runtime::do_all_coupled (
      boost::counting_iterator<unsigned> (0),
      boost::counting_iterator<unsigned> (wl.numRows ()),
      RemoveMarked<T, C> (wl),
      "remove_marked");
      
}

template <typename T, typename C>
struct RemoveMarkedStable: public RemoveMarked<T, C> {
  typedef RemoveMarked<T, C> Super_ty;

  RemoveMarkedStable (typename Super_ty::WL_ty& _wl): Super_ty (_wl) {}

  void operator () (unsigned r) {
    assert (r < Super_ty::wl.numRows ());

    typename Super_ty::WL_ty::iterator new_end =
      std::stable_partition (Super_ty::wl[r].begin (), Super_ty::wl[r].end (), IsNotMarked<T> ());

    Super_ty::wl[r].erase (new_end, Super_ty::wl[r].end ());
    
  }
};

template <typename T, typename C>
void removeMarkedStable (Galois::Runtime::PerThreadWorkList<Markable<T>, C>& wl) {

  Galois::do_all (
  // Galois::Runtime::do_all_coupled (
      boost::counting_iterator<unsigned> (0),
      boost::counting_iterator<unsigned> (wl.numRows ()),
      RemoveMarkedStable<T, C> (wl),
      "remove_marked_stable");
      
}




#endif // GALOIS_UTIL_MARKED_H

