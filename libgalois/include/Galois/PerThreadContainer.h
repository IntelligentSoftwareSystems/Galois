/** Per Thread Containers-*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * a thread local stl container for each thread
 *
 * @author <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_PERTHREADCONTAINER_H
#define GALOIS_PERTHREADCONTAINER_H

#include <vector>
#include <deque>
#include <list>
#include <set>
#include <limits>
#include <iterator>

#include <cstdio>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "Galois/gdeque.h"
#include "Galois/gstl.h"
#include "Galois/Threads.h"
#include "Galois/PriorityQueue.h"
#include "Galois/TwoLevelIterator.h"
#include "Galois/Runtime/Executor_DoAll.h"
#include "Galois/Runtime/Executor_OnEach.h"
#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Runtime/Mem.h"
#include "Galois/gIO.h"

namespace galois {

namespace {

enum GlobalPos {
  GLOBAL_BEGIN, GLOBAL_END
};

#define ADAPTOR_BASED_OUTER_ITER

// XXX: use a combination of boost::transform_iterator and
// boost::counting_iterator to implement the following OuterPerThreadWLIter
#ifdef ADAPTOR_BASED_OUTER_ITER

template<typename PerThrdCont>
struct WLindexer: 
  public std::unary_function<unsigned, typename PerThrdCont::container_type&> 
{
  typedef typename PerThrdCont::container_type Ret_ty;

  PerThrdCont* wl;

  WLindexer(): wl(NULL) {}

  WLindexer(PerThrdCont& _wl): wl(&_wl) {}

  Ret_ty& operator()(unsigned i) const {
    assert(wl != NULL);
    assert(i < wl->numRows());
    return const_cast<Ret_ty&>(wl->get(i));
  }
};

template<typename PerThrdCont>
struct TypeFactory {
  typedef typename boost::transform_iterator<WLindexer<PerThrdCont>, boost::counting_iterator<unsigned> > OuterIter;
  typedef typename std::reverse_iterator<OuterIter> RvrsOuterIter;
};


template<typename PerThrdCont>
typename TypeFactory<PerThrdCont>::OuterIter make_outer_begin(PerThrdCont& wl) {
  return boost::make_transform_iterator(
      boost::counting_iterator<unsigned>(0), WLindexer<PerThrdCont>(wl));
}

template<typename PerThrdCont>
typename TypeFactory<PerThrdCont>::OuterIter make_outer_end(PerThrdCont& wl) {
  return boost::make_transform_iterator(
      boost::counting_iterator<unsigned>(wl.numRows()), WLindexer<PerThrdCont>(wl));
}

template<typename PerThrdCont>
typename TypeFactory<PerThrdCont>::RvrsOuterIter make_outer_rbegin(PerThrdCont& wl) {
  return typename TypeFactory<PerThrdCont>::RvrsOuterIter(make_outer_end(wl));
}

template<typename PerThrdCont>
typename TypeFactory<PerThrdCont>::RvrsOuterIter make_outer_rend(PerThrdCont& wl) {
  return typename TypeFactory<PerThrdCont>::RvrsOuterIter(make_outer_begin(wl));
}

#else

template<typename PerThrdCont>
class OuterPerThreadWLIter: public boost::iterator_facade<
  OuterPerThreadWLIter<PerThrdCont>, 
  typename PerThrdCont::container_type, 
  boost::random_access_traversal_tag> {




  using container_type = typename PerThrdCont::container_type;
  using Diff_ty = ptrdiff_t;

  friend class boost::iterator_core_access;

  PerThrdCont* workList;
  // using Diff_ty due to reverse iterator, whose 
  // end is -1, and,  begin is numRows - 1
  Diff_ty row;

  void assertInRange() const {
    assert((row >= 0) && (row < workList->numRows()));
  }

  // container_type& getWL() {
    // assertInRange();
    // return (*workList)[row];
  // }

  container_type& getWL() const {
    assertInRange();
    return (*workList)[row];
  }

public:

  OuterPerThreadWLIter(): workList(NULL), row(0) {}

  OuterPerThreadWLIter(PerThrdCont& wl, const GlobalPos& pos)
    : workList(&wl), row(0) {

    switch (pos) {
      case GLOBAL_BEGIN:
        row = 0;
        break;
      case GLOBAL_END:
        row = wl.numRows();
        break;
      default:
        std::abort();
    }
  }

  container_type& dereference (void) const { 
    return getWL ();
  }

  // const container_type& dereference (void) const {
    // getWL ();
  // }


  void increment (void) {
    ++row;
  }

  void decrement (void) {
    --row;
  }

  bool equal (const OuterPerThreadWLIter& that) const {
    assert (this->workList == that.workList);
    return this->row == that.row;
  }

  void advance (ptrdiff_t n) {
    row += n;
  }

  Diff_ty distance_to (const OuterPerThreadWLIter& that) const {
    assert (this->workList == that.workList);
    return that.row - this->row;
  }
};

template<typename PerThrdCont>
OuterPerThreadWLIter<PerThrdCont> make_outer_begin(PerThrdCont& wl) {
  return OuterPerThreadWLIter<PerThrdCont>(wl, GLOBAL_BEGIN);
}

template<typename PerThrdCont>
OuterPerThreadWLIter<PerThrdCont> make_outer_end(PerThrdCont& wl) {
  return OuterPerThreadWLIter<PerThrdCont>(wl, GLOBAL_END);
}

template<typename PerThrdCont>
std::reverse_iterator<OuterPerThreadWLIter<PerThrdCont> > 
make_outer_rbegin(PerThrdCont& wl) {
  typedef typename std::reverse_iterator<OuterPerThreadWLIter<PerThrdCont> > Ret_ty;
  return Ret_ty(make_outer_end(wl));
}

template<typename PerThrdCont>
std::reverse_iterator<OuterPerThreadWLIter<PerThrdCont> > 
make_outer_rend(PerThrdCont& wl) {
  typedef typename std::reverse_iterator<OuterPerThreadWLIter<PerThrdCont> > Ret_ty;
  return Ret_ty(make_outer_begin(wl));
}

#endif

} // end namespace 


template<typename Cont_tp> 
class PerThreadContainer {
public:
  typedef Cont_tp container_type;
  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::pointer pointer;
  typedef typename container_type::size_type size_type;

  typedef typename container_type::iterator local_iterator;
  typedef typename container_type::const_iterator local_const_iterator;
  typedef typename container_type::reverse_iterator local_reverse_iterator;
  typedef typename container_type::const_reverse_iterator local_const_reverse_iterator;

  typedef PerThreadContainer This_ty;

#ifdef ADAPTOR_BASED_OUTER_ITER
  typedef typename TypeFactory<This_ty>::OuterIter OuterIter;
  typedef typename TypeFactory<This_ty>::RvrsOuterIter RvrsOuterIter;
#else
  typedef OuterPerThreadWLIter<This_ty> OuterIter;
  typedef typename std::reverse_iterator<OuterIter> RvrsOuterIter;
#endif
  typedef typename galois::ChooseStlTwoLevelIterator<OuterIter, typename container_type::iterator>::type global_iterator;
  typedef typename galois::ChooseStlTwoLevelIterator<OuterIter, typename container_type::const_iterator>::type global_const_iterator;
  typedef typename galois::ChooseStlTwoLevelIterator<RvrsOuterIter, typename container_type::reverse_iterator>::type global_reverse_iterator;
  typedef typename galois::ChooseStlTwoLevelIterator<RvrsOuterIter, typename container_type::const_reverse_iterator>::type global_const_reverse_iterator;

  typedef global_iterator iterator;
  typedef global_const_iterator const_iterator;
  typedef global_reverse_iterator reverse_iterator;
  typedef global_const_reverse_iterator const_reverse_iterator;
private:

  // XXX: for testing only

#if 0
  struct FakePTS {
    std::vector<container_type*> v;

    FakePTS () { 
      v.resize (size ());
    }

    container_type** getLocal () const {
      return getRemote (galois::Runtime::LL::getTID ());
    }

    container_type** getRemote (size_t i) const {
      assert (i < v.size ());
      return const_cast<container_type**> (&v[i]);
    }

    size_t size () const { return galois::Runtime::LL::getMaxThreads(); }

  };
#endif
  // typedef FakePTS PerThrdCont_ty;
  typedef galois::Substrate::PerThreadStorage<container_type*> PerThrdCont_ty;
  PerThrdCont_ty perThrdCont;

  void destroy() {
    for (unsigned i = 0; i < perThrdCont.size(); ++i) {
      delete *perThrdCont.getRemote(i);
      *perThrdCont.getRemote(i) = NULL;
    }
  }

protected:
  PerThreadContainer(): perThrdCont() {
    for (unsigned i = 0; i < perThrdCont.size(); ++i) {
      *perThrdCont.getRemote(i) = NULL;
    }
  }

  template <typename... Args>
  void init(Args&&... args) {
    for (unsigned i = 0; i < perThrdCont.size(); ++i) {
      *perThrdCont.getRemote(i) = new container_type(std::forward<Args> (args)...);
    }
  }

  ~PerThreadContainer() { 
    clear_all_parallel ();
    destroy();
  }

public:
  unsigned numRows() const { return perThrdCont.size(); }

  container_type& get() { return **(perThrdCont.getLocal()); }

  const container_type& get() const { return **(perThrdCont.getLocal()); }

  container_type& get(unsigned i) { return **(perThrdCont.getRemote(i)); }

  const container_type& get(unsigned i) const { return **(perThrdCont.getRemote(i)); }

  container_type& operator [](unsigned i) { return get(i); }

  const container_type& operator [](unsigned i) const { return get(i); }


  global_iterator begin_all() { 
    return galois::stl_two_level_begin(
        make_outer_begin(*this), make_outer_end(*this)); 
  }

  global_iterator end_all() { 
    return galois::stl_two_level_end(
        make_outer_begin(*this), make_outer_end(*this)); 
  }

  global_const_iterator begin_all() const { 
    return cbegin_all ();
  }

  global_const_iterator end_all() const { 
    return cend_all ();
  }

  // for compatibility with Range.h
  global_iterator begin () { return begin_all (); }

  global_iterator end () { return end_all (); }

  global_const_iterator begin () const { return begin_all (); }

  global_const_iterator end () const { return end_all (); }

  global_const_iterator cbegin () const { return cbegin_all (); }

  global_const_iterator cend () const { return cend_all (); }

  global_const_iterator cbegin_all() const { 
    return galois::stl_two_level_cbegin(
        make_outer_begin(*this), make_outer_end(*this));
  }

  global_const_iterator cend_all() const { 
    return galois::stl_two_level_cend(
        make_outer_begin(*this), make_outer_end(*this));
  }

  global_reverse_iterator rbegin_all() { 
    return galois::stl_two_level_rbegin(
        make_outer_rbegin(*this), make_outer_rend(*this)); 
  }

  global_reverse_iterator rend_all() { 
    return galois::stl_two_level_rend(
        make_outer_rbegin(*this), make_outer_rend(*this)); 
  }

  global_const_reverse_iterator rbegin_all() const { 
    return crbegin_all ();
  }

  global_const_reverse_iterator rend_all() const { 
    return crend_all ();
  }

  global_const_reverse_iterator crbegin_all() const { 
    return galois::stl_two_level_crbegin(
        make_outer_rbegin(*this), make_outer_rend(*this));
  }

  global_const_reverse_iterator crend_all() const { 
    return galois::stl_two_level_crend(
        make_outer_rbegin(*this), make_outer_rend(*this));
  }

  local_iterator local_begin () { return get ().begin (); }
  local_iterator local_end () { return get ().end (); }

  // legacy STL
  local_const_iterator local_begin () const { return get ().begin (); }
  local_const_iterator local_end () const { return get ().end (); }

  local_const_iterator local_cbegin () const { return get ().cbegin (); }
  local_const_iterator local_cend () const { return get ().cend (); }

  local_reverse_iterator local_rbegin () { return get ().rbegin (); }
  local_reverse_iterator local_rend () { return get ().rend (); }

  local_const_reverse_iterator local_crbegin () const { return get ().crbegin (); }
  local_const_reverse_iterator local_crend () const { return get ().crend (); }

  size_type size_all() const {
    size_type sz = 0;

    for (unsigned i = 0; i < perThrdCont.size(); ++i) {
      sz += get(i).size();
    }

    return sz;
  }

  // XXX: disabling because of per thread memory allocators 
  // void clear_all() {
    // for (unsigned i = 0; i < perThrdCont.size(); ++i) {
      // get(i).clear();
    // }
  // }

  void clear_all_parallel (void) {

    Runtime::on_each_impl(
        [this] (const unsigned tid, const unsigned numT) {
          get ().clear ();
        });


    // XXX: following implementation causes problems with allocators
    // that allocate per thread separately
    // auto r = galois::Runtime::makeStandardRange (
        // boost::counting_iterator<unsigned> (0),
        // boost::counting_iterator<unsigned> (perThrdCont.size ()));
// 
    // galois::Runtime:: do_all_impl (
        // r, 
        // [this] (unsigned i) {
          // get (i).clear ();
        // },
        // "clear_all", 
        // false);
    
  }

  bool empty_all() const {
    bool res = true;
    for (unsigned i = 0; i < perThrdCont.size(); ++i) {
      res = res && get(i).empty();
    }

    return res;
  }

  template <typename Range, typename Ret>
  void fill_parallel (const Range& range, Ret (container_type::*pushFn) (const value_type&) = &container_type::push_back) {
    galois::Runtime::do_all_impl (
        range,
        [this, pushFn] (const typename Range::value_type& v) {
          container_type& my = get ();
          (my.*pushFn) (v);
          // (get ().*pushFn)(v);
        },
        "fill_parallel", 
        false);
  }

  // // TODO: fill parallel
  // template<typename Iter, typename R>
  // void fill_serial(Iter begin, Iter end,
      // R(container_type::*pushFn)(const value_type&) = &container_type::push_back) {
// 
    // const unsigned P = galois::getActiveThreads();
// 
    // typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;
// 
    // // integer division, where we want to round up. So adding P-1
    // Diff_ty block_size = (std::distance(begin, end) + (P-1) ) / P;
// 
    // assert(block_size >= 1);
// 
    // Iter block_begin = begin;
// 
    // for (unsigned i = 0; i < P; ++i) {
// 
      // Iter block_end = block_begin;
// 
      // if (std::distance(block_end, end) < block_size) {
        // block_end = end;
// 
      // } else {
        // std::advance(block_end, block_size);
      // }
// 
      // for (; block_begin != block_end; ++block_begin) {
        // // workList[i].push_back(Marked<Value_ty>(*block_begin));
        // ((*this)[i].*pushFn)(value_type(*block_begin));
      // }
// 
      // if (block_end == end) {
        // break;
      // }
    // }
  // }
};

template<typename T>
class PerThreadVector: public PerThreadContainer<typename gstl::template Vector<T> > {
public:
  typedef typename gstl::template Pow2Alloc<T>  Alloc_ty;
  typedef typename gstl::template Vector<T>  container_type;

protected:
  typedef PerThreadContainer<container_type> Super_ty;

  Alloc_ty alloc;

public:
  PerThreadVector(): Super_ty(), alloc() {
    Super_ty::init(alloc);
  }

  void reserve_all(size_t sz) {
    size_t numT = galois::getActiveThreads();
    size_t perT = (sz + numT - 1) / numT; // round up

    for (unsigned i = 0; i < numT; ++i) {
      Super_ty::get(i).reserve(perT);
    }
  }
};


template<typename T>
class PerThreadDeque: 
  public PerThreadContainer<typename gstl::template Deque<T> > {

public:
  typedef typename gstl::template Pow2Alloc<T>  Alloc_ty;

protected:
  typedef typename gstl::template Deque<T>  container_type;
  typedef PerThreadContainer<container_type> Super_ty;

  Alloc_ty alloc;

public:
  PerThreadDeque(): Super_ty(), alloc() {
    Super_ty::init(alloc);
  }
};

template <typename T, unsigned ChunkSize=64>
class PerThreadGdeque: public PerThreadContainer<galois::gdeque<T, ChunkSize> > {

  using Super_ty = PerThreadContainer<galois::gdeque<T, ChunkSize> >;
  
public:

  PerThreadGdeque (): Super_ty () {
    Super_ty::init ();
  }
};

template<typename T>
class PerThreadList:
  public PerThreadContainer<typename gstl::template List<T> > {

public:
  typedef typename gstl::template FixedSizeAlloc<T>  Alloc_ty;

protected:
  typedef typename gstl::template List<T>  container_type;
  typedef PerThreadContainer<container_type> Super_ty;

  Alloc_ty alloc;

public:
  PerThreadList(): Super_ty(), alloc() {
    Super_ty::init(alloc);
  }
};

template<typename T, typename C=std::less<T> >
class PerThreadSet: 
  public PerThreadContainer<typename gstl::template Set<T, C> > {

public:
  typedef typename gstl::template FixedSizeAlloc<T>  Alloc_ty;

protected:
  typedef typename gstl::template Set<T, C>  container_type;
  typedef PerThreadContainer<container_type> Super_ty;

  Alloc_ty alloc;

public:
  explicit PerThreadSet(const C& cmp = C()): Super_ty(), alloc() {
    Super_ty::init(cmp, alloc);
  }

  typedef typename Super_ty::global_const_iterator global_const_iterator;
  typedef typename Super_ty::global_const_reverse_iterator global_const_reverse_iterator;

  // hiding non-const (and const) versions in Super_ty
  global_const_iterator begin_all() const { return Super_ty::cbegin_all(); }
  global_const_iterator end_all() const { return Super_ty::cend_all(); }

  // hiding non-const (and const) versions in Super_ty
  global_const_reverse_iterator rbegin_all() const { return Super_ty::crbegin_all(); }
  global_const_reverse_iterator rend_all() const { return Super_ty::crend_all(); }
};


template<typename T, typename C=std::less<T> >
class PerThreadMinHeap:
  public PerThreadContainer<typename gstl::template PQ<T, C> > {

public:
  typedef typename gstl::template Pow2Alloc<T>  Alloc_ty;

protected:
  typedef typename gstl::template Vector<T>  Vec_ty;
  typedef typename gstl::template PQ<T, C>  container_type;
  typedef PerThreadContainer<container_type> Super_ty;

  Alloc_ty alloc;

public:
  explicit PerThreadMinHeap(const C& cmp = C()): Super_ty(), alloc() {
    Super_ty::init(cmp, Vec_ty(alloc));
  }

  typedef typename Super_ty::global_const_iterator global_const_iterator;
  typedef typename Super_ty::global_const_reverse_iterator global_const_reverse_iterator;

  // hiding non-const (and const) versions in Super_ty
  global_const_iterator begin_all() const { return Super_ty::cbegin_all(); }
  global_const_iterator end_all() const { return Super_ty::cend_all(); }

  // hiding non-const (and const) versions in Super_ty
  global_const_reverse_iterator rbegin_all() const { return Super_ty::crbegin_all(); }
  global_const_reverse_iterator rend_all() const { return Super_ty::crend_all(); }
};


} // end namespace galois
#endif // GALOIS_PERTHREADCONTAINER_H
