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

namespace GaloisRuntime {

template <typename T, template <typename _u, typename _a> class C, typename AllocTy=std::allocator<T> > 
struct PerThreadWorkList {

  typedef C<T, AllocTy> ContTy;
  typedef GaloisRuntime::PerCPU<ContTy> PerThrdContTy;

  PerThrdContTy perThrdCont;

public:
  typedef typename ContTy::value_type value_type;
  typedef typename ContTy::iterator iterator;
  typedef typename ContTy::const_iterator const_iterator;
  typedef typename ContTy::reference reference;
  typedef typename ContTy::pointer pointer;
  typedef typename ContTy::size_type size_type;


  PerThreadWorkList () {}

  PerThreadWorkList (const ContTy& refCont): perThrdCont (refCont) {}

  unsigned numRows () const { return perThrdCont.size (); }

  ContTy& operator [] (unsigned i) { return perThrdCont.get (i); }

  const ContTy& operator [] (unsigned i) const { return perThrdCont.get (i); }

  ContTy& get () { return perThrdCont.get (); }

  const ContTy& get () const { return perThrdCont.get (); }

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

private:

};

template <typename T>
struct PerThreadWLfactory {

  // TODO: change this to a better one
  typedef std::allocator<T> AllocTy;

  typedef PerThreadWorkList<T, std::vector, AllocTy> PerThreadVector;


};

}




#endif // GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_
