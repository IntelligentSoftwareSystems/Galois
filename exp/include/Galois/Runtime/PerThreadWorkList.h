/** Per Thread workLists-*- C++ -*-
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
 * @section Description
 *
 * a thread local stl container for each thread
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_
#define GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_

#include <vector>
#include <limits>
#include <iostream>

#include <cstdio>

#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/Threads.h"

namespace GaloisRuntime {

template <typename T, typename Cont_tp> 
struct PerThreadWorkList {

  typedef Cont_tp Cont_ty;
  typedef GaloisRuntime::PerCPU<Cont_ty> PerThrdCont_ty;

  PerThrdCont_ty perThrdCont;

public:
  typedef typename Cont_ty::value_type value_type;
  typedef typename Cont_ty::iterator iterator;
  typedef typename Cont_ty::const_iterator const_iterator;
  typedef typename Cont_ty::reference reference;
  typedef typename Cont_ty::pointer pointer;
  typedef typename Cont_ty::size_type size_type;


  PerThreadWorkList () {}

  PerThreadWorkList (const Cont_ty& refCont): perThrdCont (refCont) {}

  unsigned numRows () const { return perThrdCont.size (); }

  Cont_ty& operator [] (unsigned i) { return perThrdCont.get (i); }

  const Cont_ty& operator [] (unsigned i) const { return perThrdCont.get (i); }

  Cont_ty& get () { return perThrdCont.get (); }

  const Cont_ty& get () const { return perThrdCont.get (); }

  iterator begin (unsigned i) { return perThrdCont.get (i).begin (); }
  const_iterator begin (unsigned i) const { return perThrdCont.get (i).begin (); }

  iterator end (unsigned i) { return perThrdCont.get (i).end (); }
  const_iterator end (unsigned i) const { return perThrdCont.get (i).end (); }


  size_type size () const { return perThrdCont.get ().size (); }
  size_type size (unsigned i) const { return perThrdCont.get (i).size (); }

  void clear () { perThrdCont.get ().clear (); }
  void clear (unsigned i) { perThrdCont.get (i).clear (); }

  bool empty () const { return perThrdCont.get ().empty (); }
  bool empty (unsigned i) const { return perThrdCont.get (i).empty (); }

  size_type size_all () const {
    size_type sz = 0;

    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      sz += size (i);
    }

    return sz;
  }


  void clear_all () {
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      clear (i);
    }
  }

  bool empty_all () const {
    bool res = true;
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      res = res && empty (i);
    }

    return res;
  }

  template <typename Iter, typename R>
  void fill_init (Iter begin, Iter end,
      R (Cont_ty::*pushFn) (const value_type&)=&Cont_ty::push_back) {

    const unsigned P = Galois::getActiveThreads ();

    typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;

    // integer division, where we want to round up. So adding P-1
    Diff_ty block_size = (std::distance (begin, end) + (P-1) ) / P;

    assert (block_size >= 1);

    Iter block_begin = begin;

    for (unsigned i = 0; i < P; ++i) {

      Iter block_end = block_begin;

      if (std::distance (block_end, end) < block_size) {
        block_end = end;

      } else {
        std::advance (block_end, block_size);
      }

      for (; block_begin != block_end; ++block_begin) {
        // workList[i].push_back (Marked<Value_ty> (*block_begin));
        ((*this)[i].*pushFn) (value_type (*block_begin));
      }

      if (block_end == end) {
        break;
      }
    }
  }

private:

};

template <typename T>
struct PerThreadWLfactory {

  // TODO: change this to a better one
  typedef std::allocator<T> AllocTy;

  typedef PerThreadWorkList<T, std::vector<T, AllocTy> > PerThreadVector;


};

}




#endif // GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_
