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
 * @author <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_
#define GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_

#include <vector>
#include <deque>
#include <set>
#include <limits>
#include <iostream>

#include <cstdio>

#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/TwoLevelIteratorA.h"

namespace GaloisRuntime {

template <typename T, typename Cont_tp> 
class PerThreadWorkList {

public:
  typedef Cont_tp Cont_ty;
  typedef typename Cont_ty::value_type value_type;
  typedef typename Cont_ty::reference reference;
  typedef typename Cont_ty::pointer pointer;
  typedef typename Cont_ty::size_type size_type;

  typedef typename Cont_ty::iterator local_iterator;
  typedef typename Cont_ty::const_iterator local_const_iterator;
  typedef typename Cont_ty::reverse_iterator local_reverse_iterator;
  typedef typename Cont_ty::const_reverse_iterator local_const_reverse_iterator;

  typedef PerThreadWorkList<T, Cont_tp> This_ty;

  typedef typename intern::ChooseIter<This_ty, typename Cont_tp::iterator>::type global_iterator;
  typedef typename intern::ChooseIter<This_ty, typename Cont_tp::const_iterator>::type global_const_iterator;
  typedef typename intern::ChooseIter<This_ty, typename Cont_tp::reverse_iterator>::type global_reverse_iterator;
  typedef typename intern::ChooseIter<This_ty, typename Cont_tp::const_reverse_iterator>::type global_const_reverse_iterator;

protected:
  typedef GaloisRuntime::PerCPU<Cont_ty*> PerThrdCont_ty;
  PerThrdCont_ty perThrdCont;

  PerThreadWorkList (): perThrdCont (NULL) {}
  ~PerThreadWorkList () {}


public:
  unsigned numRows () const { return perThrdCont.size (); }

  Cont_ty& get () { return *(perThrdCont.get ()); }

  const Cont_ty& get () const { return *(perThrdCont.get ()); }

  Cont_ty& get (unsigned i) { return *(perThrdCont.get (i)); }

  const Cont_ty& get (unsigned i) const { return *(perThrdCont.get (i)); }

  Cont_ty& operator [] (unsigned i) { return get (i); }

  const Cont_ty& operator [] (unsigned i) const { return get (i); }

  global_iterator begin_all () { return intern::make_begin (*this, local_iterator ()); }
  global_iterator end_all () { return intern::make_end (*this, local_iterator ()); }

  global_const_iterator begin_all () const { return intern::make_begin (*this, local_const_iterator ()); }
  global_const_iterator end_all () const { return intern::make_end (*this, local_const_iterator ()); }

  global_reverse_iterator rbegin_all () { return intern::make_begin (*this, local_reverse_iterator ()); }
  global_reverse_iterator rend_all () { return intern::make_end (*this, local_reverse_iterator ()); }

  global_const_reverse_iterator rbegin_all () const { return intern::make_begin (*this, local_const_reverse_iterator ()); }
  global_const_reverse_iterator rend_all () const { return intern::make_end (*this, local_const_reverse_iterator ()); }

  size_type size_all () const {
    size_type sz = 0;

    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      sz += get (i).size ();
    }

    return sz;
  }


  void clear_all () {
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      get (i).clear ();
    }
  }

  bool empty_all () const {
    bool res = true;
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      res = res && get (i).empty ();
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

};

template <typename T>
struct PerThreadWLfactory {

  typedef std::allocator<T> StlAlloc;

  typedef PerThreadWorkList<T, std::vector<T, StlAlloc> > PerThreadStlVector;


  typedef MM::SimpleBumpPtrWithMallocFallback<MM::FreeListHeap<MM::SystemBaseAlloc> > BasicHeap;

  typedef MM::ThreadAwarePrivateHeap<BasicHeap> PerThreadHeap;

  typedef MM::ExternRefGaloisAllocator<T, PerThreadHeap> PerThreadAllocator;

};


template <typename T>
class PerThreadVector: 
  public PerThreadWorkList<T, std::vector<T, typename PerThreadWLfactory<T>::PerThreadAllocator> > {

public:
  typedef typename PerThreadWLfactory<T>::PerThreadHeap Heap_ty;
  typedef typename PerThreadWLfactory<T>::PerThreadAllocator Alloc_ty;
  typedef std::vector<T, Alloc_ty> Cont_ty;

protected:
  typedef PerThreadWorkList<T, Cont_ty> Super_ty;


  Heap_ty heap;


public:
  PerThreadVector () {
    Alloc_ty alloc (&heap);

    for (unsigned i = 0; i < Super_ty::perThrdCont.size (); ++i) {
      Super_ty::perThrdCont.get (i) = new Cont_ty (alloc);
    }

  }

  ~PerThreadVector () {
    for (unsigned i = 0; i < Super_ty::perThrdCont.size (); ++i) {
      delete Super_ty::perThrdCont.get (i);
      Super_ty::perThrdCont.get (i) = NULL;
    }
  }
  

};


template <typename T>
class PerThreadDeque: 
  public PerThreadWorkList<T, std::deque<T, typename PerThreadWLfactory<T>::PerThreadAllocator> > {

public:
  typedef typename PerThreadWLfactory<T>::PerThreadHeap Heap_ty;
  typedef typename PerThreadWLfactory<T>::PerThreadAllocator Alloc_ty;
  typedef std::deque<T, Alloc_ty> Cont_ty;

protected:
  typedef PerThreadWorkList<T, Cont_ty> Super_ty;


  Heap_ty heap;


public:
  PerThreadDeque () {
    Alloc_ty alloc (&heap);

    for (unsigned i = 0; i < Super_ty::perThrdCont.size (); ++i) {
      Super_ty::perThrdCont.get (i) = new Cont_ty (alloc);
    }

  }

  ~PerThreadDeque () {
    for (unsigned i = 0; i < Super_ty::perThrdCont.size (); ++i) {
      delete Super_ty::perThrdCont.get (i);
      Super_ty::perThrdCont.get (i) = NULL;
    }
  }
  

};

template <typename T, typename C=std::less<T> >
class PerThreadSet: 
  public PerThreadWorkList<T, std::set<T, C, typename PerThreadWLfactory<T>::PerThreadAllocator> > {

public:
  typedef typename PerThreadWLfactory<T>::PerThreadHeap Heap_ty;
  typedef typename PerThreadWLfactory<T>::PerThreadAllocator Alloc_ty;
  typedef std::set<T, C, Alloc_ty> Cont_ty;

protected:
  typedef PerThreadWorkList<T, Cont_ty> Super_ty;


  Heap_ty heap;


public:
  explicit PerThreadSet (const C& cmp=C ()) {
    Alloc_ty alloc (&heap);

    for (unsigned i = 0; i < Super_ty::perThrdCont.size (); ++i) {
      Super_ty::perThrdCont.get (i) = new Cont_ty (cmp, alloc);
    }

  }

  ~PerThreadSet () {
    for (unsigned i = 0; i < Super_ty::perThrdCont.size (); ++i) {
      delete Super_ty::perThrdCont.get (i);
      Super_ty::perThrdCont.get (i) = NULL;
    }
  }
  

};

}




#endif // GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_
