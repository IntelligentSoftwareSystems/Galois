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

#include "Galois/Threads.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/TwoLevelIterator.h"

namespace Galois {
namespace Runtime {

namespace HIDDEN {

enum GlobalPos {
  GLOBAL_BEGIN, GLOBAL_END
};

// #define USE_CUSTOM_TWO_LEVEL_ITER

// TODO: choose one at the end
#ifndef USE_CUSTOM_TWO_LEVEL_ITER

// TODO: use a combination of boost::transform_iterator and
// boost::counting_iterator to implement the following OuterPerThreadWLIter
template <typename PerThrdWL>
class OuterPerThreadWLIter: public std::iterator<std::random_access_iterator_tag, typename PerThrdWL::Cont_ty> {

  typedef typename PerThrdWL::Cont_ty Cont_ty;
  typedef std::iterator<std::random_access_iterator_tag, Cont_ty> Super_ty;
  typedef typename Super_ty::difference_type Diff_ty;

  PerThrdWL* workList;
  // using Diff_ty due to reverse iterator, whose 
  // end is -1, and,  begin is numRows - 1
  Diff_ty row;

  void assertInRange () const {
    assert ((row >= 0) && (row < workList->numRows ()));
  }

  Cont_ty& getWL () {
    assertInRange ();
    return (*workList)[row];
  }

  const Cont_ty& getWL () const {
    assertInRange ();
    return (*workList)[row];
  }


public:

  OuterPerThreadWLIter (): Super_ty (), workList (NULL), row (0) {}

  OuterPerThreadWLIter (PerThrdWL& wl, const GlobalPos& pos)
    : Super_ty (), workList (&wl), row (0) {

    switch (pos) {
      case GLOBAL_BEGIN:
        row = 0;
        break;
      case GLOBAL_END:
        row = wl.numRows ();
        break;
      default:
        std::abort ();
    }
  }

  typename Super_ty::reference operator * () { return getWL (); }

  const typename Super_ty::reference operator * () const { return getWL (); }

  typename Super_ty::pointer operator -> () { return &(getWL ()); }

  const typename Super_ty::value_type* operator -> () const { return &(getWL ()); }

  OuterPerThreadWLIter& operator ++ () {
    ++row;
    return *this;
  }

  OuterPerThreadWLIter operator ++ (int) {
    OuterPerThreadWLIter tmp (*this);
    operator ++ ();
    return tmp;
  }

  OuterPerThreadWLIter& operator -- () {
    --row;
    return *this;
  }

  OuterPerThreadWLIter operator -- (int) {
    OuterPerThreadWLIter tmp (*this);
    operator -- ();
    return tmp;
  }

  OuterPerThreadWLIter& operator += (Diff_ty d) {
    row = unsigned (Diff_ty (row) + d);
    return *this;
  }

  OuterPerThreadWLIter& operator -= (Diff_ty d) {
    row = unsigned (Diff_ty (row) - d);
    return *this;
  }

  friend OuterPerThreadWLIter operator + (const OuterPerThreadWLIter& it, Diff_ty d) {
    OuterPerThreadWLIter tmp (it);
    tmp += d;
    return tmp;
  }

  friend OuterPerThreadWLIter operator + (Diff_ty d, const OuterPerThreadWLIter& it) {
    return it + d;
  }

  friend OuterPerThreadWLIter operator - (const OuterPerThreadWLIter& it, Diff_ty d) {
    OuterPerThreadWLIter tmp (it);
    tmp -= d;
    return tmp;
  }

  friend Diff_ty operator - (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {
    return Diff_ty (left.row) - Diff_ty (right.row);
  }

  typename Super_ty::reference operator [] (Diff_ty d) {
    return *((*this) + d);
  }

  friend bool operator == (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    assert (left.workList == right.workList);
    return (left.row == right.row);
  }

  friend bool operator != (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {
    return !(left == right);
  }

  friend bool operator < (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    assert (left.workList == right.workList);

    return (left.row < right.row);
  }

  friend bool operator <= (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    return (left == right) || (left < right);
  }

  friend bool operator > (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    return !(left <= right);
  }

  friend bool operator >= (const OuterPerThreadWLIter& left, const OuterPerThreadWLIter& right) {

    return !(left < right);
  }

};


template <typename PerThrdWL>
OuterPerThreadWLIter<PerThrdWL> make_outer_begin (PerThrdWL& wl) {
  return OuterPerThreadWLIter<PerThrdWL> (wl, GLOBAL_BEGIN);
}

template <typename PerThrdWL>
OuterPerThreadWLIter<PerThrdWL> make_outer_end (PerThrdWL& wl) {
  return OuterPerThreadWLIter<PerThrdWL> (wl, GLOBAL_END);
}

template <typename PerThrdWL>
std::reverse_iterator<OuterPerThreadWLIter<PerThrdWL> > 
  make_outer_rbegin (PerThrdWL& wl) {
  typedef std::reverse_iterator<OuterPerThreadWLIter<PerThrdWL> > Ret_ty;
  return Ret_ty (make_outer_end (wl));
}

template <typename PerThrdWL>
std::reverse_iterator<OuterPerThreadWLIter<PerThrdWL> > 
  make_outer_rend (PerThrdWL& wl) {
  typedef std::reverse_iterator<OuterPerThreadWLIter<PerThrdWL> > Ret_ty;
  return Ret_ty (make_outer_begin (wl));
}


#else 

template <typename PerThrdWL, typename Iter, bool is_reverse_tp>
class TwoLevelIterBase {

protected:
  PerThrdWL* workList;
  unsigned row;
  Iter curr;

  PerThrdWL& getWL () const { 
    assert (workList != NULL);
    return *workList; 
  }

  inline Iter localBegin () const { 
    return getWL ()[row].begin ();
  }

  inline Iter localEnd () const {
    return getWL ()[row].end  ();
  }

  void nextRow () {
    ++row;
    assert (row < getWL ().numRows ());
    curr = localBegin ();
  }

  void prevRow () {
    assert (row > 0);
    --row;
    curr = localEnd ();
  }

  TwoLevelIterBase (): workList (NULL), row (0), curr () {}

  TwoLevelIterBase (PerThrdWL& wl, const GlobalPos& pos)
    : workList (&wl), row (0), curr () {

    switch (pos) {
      case GLOBAL_BEGIN: 
        row = 0;
        curr = localBegin ();
        break;
      case GLOBAL_END:
        row = wl.numRows () - 1;
        curr = localEnd ();
        break;
      default:
        std::abort ();
    }
  }

};

template <typename PerThrdWL, typename Iter>
class TwoLevelIterBase<PerThrdWL, Iter, true> {

protected:
  typedef std::iterator_traits<Iter> Traits;

  PerThrdWL* workList;
  unsigned row;
  Iter curr;

  PerThrdWL& getWL () const { 
    assert (workList != NULL);
    return *workList; 
  }

  inline Iter localBegin () const { 
    return getWL ()[row].rbegin ();
  }

  inline Iter localEnd () const {
    return getWL ()[row].rend ();
  }

  void nextRow () {
    assert (row > 0);
    --row;
    curr = localBegin ();
  }

  void prevRow () {
    ++row;
    assert (row < getWL ().numRows ());
    curr = localEnd ();
  }

  TwoLevelIterBase (): workList (NULL), row (0), curr () {}

  TwoLevelIterBase (PerThrdWL& wl, const GlobalPos& pos)
    : workList (&wl), row (0), curr () {

    switch (pos) {
      case GLOBAL_BEGIN: 
        row = wl.numRows () - 1;
        curr = localBegin (row);
        break;
      case GLOBAL_END:
        row = 0;
        curr = localEnd (row);
        break;
      default:
        std::abort ();
    }
  }
};

template <typename PerThrdWL, typename Iter>
struct IsRvrs {
  static const bool VAL = false;
};

template <typename PerThrdWL>
struct IsRvrs<PerThrdWL, typename PerThrdWL::Cont_ty::reverse_iterator> {
  static const bool VAL = true;
};

template <typename PerThrdWL>
struct IsRvrs<PerThrdWL, typename PerThrdWL::Cont_ty::const_reverse_iterator> {
  static const bool VAL = true;
};

template <typename PerThrdWL, typename Iter>
class TwoLevelFwdIter: 
  public std::iterator_traits<Iter>, 
  public TwoLevelIterBase<PerThrdWL, Iter, IsRvrs<PerThrdWL, Iter>::VAL> {

protected:

  typedef std::iterator_traits<Iter> Traits;
  typedef TwoLevelIterBase<PerThrdWL, Iter, IsRvrs<PerThrdWL, Iter>::VAL> Base;

  inline bool atBegin () const {
    return Base::curr == Base::localBegin ();
  }

  inline bool atEnd () const {
    return Base::curr == Base::localEnd ();
  }

  void seekValidBegin () {
    while ((Base::row < (Base::workList->numRows () - 1)) &&  atEnd ()) {
      Base::nextRow ();
    } 
  }

  void step_forward () {

    assert (!atEnd ());
    ++Base::curr;
    
    if (atEnd ()) {
      seekValidBegin ();
    }
  }

  bool is_equal (const TwoLevelFwdIter& that) const {
    assert (this->workList == that.workList);

    return (this->row == that.row) 
      && (this->curr == that.curr);
  }


public:

  TwoLevelFwdIter (): Base () {}

  TwoLevelFwdIter (PerThrdWL& wl, const GlobalPos& pos): Base (wl, pos) {
    // Base::curr = Base::localBegin ();
    seekValidBegin ();
  }

  typename Traits::reference operator * () { 
    return *Base::curr;
  }

  typename Traits::pointer operator -> () {
    return Base::curr.operator -> ();
  }

  TwoLevelFwdIter& operator ++ () {
    step_forward ();
    return *this;
  }

  TwoLevelFwdIter operator ++ (int) {
    TwoLevelFwdIter tmp (*this);
    step_forward ();
    return tmp;
  }
    
  friend bool operator == (const TwoLevelFwdIter& left, const TwoLevelFwdIter& right) {
    return left.is_equal (right);
  }

  friend bool operator != (const TwoLevelFwdIter& left, const TwoLevelFwdIter& right) {
    return !left.is_equal (right);
  }


};

template <typename PerThrdWL, typename Iter>
class TwoLevelBiDirIter: public TwoLevelFwdIter<PerThrdWL, Iter> {

protected:
  typedef TwoLevelFwdIter<PerThrdWL, Iter> FwdBase;


  void step_backward () {
    while (FwdBase::row > 0 && FwdBase::atBegin ()) {
      FwdBase::prevRow ();
    }

    if (!FwdBase::atBegin ()) {
      --FwdBase::curr;
    }
  }

public:

  TwoLevelBiDirIter (): FwdBase () {}

  TwoLevelBiDirIter (PerThrdWL& wl, const GlobalPos& pos): FwdBase (wl, pos) {}


  TwoLevelBiDirIter& operator -- () {
    step_backward ();
    return *this;
  }

  TwoLevelBiDirIter operator -- (int) {
    TwoLevelBiDirIter tmp (*this);
    step_backward ();
    return tmp;
  }
};


template <typename PerThrdWL, typename Iter>
class TwoLevelRandIter: public TwoLevelBiDirIter<PerThrdWL, Iter> {

protected:
  typedef TwoLevelBiDirIter<PerThrdWL, Iter> BiDirBase;

  typedef typename BiDirBase::Traits::difference_type Diff_ty;

  void jump_forward (const Diff_ty d) {
    if (d < 0) {
      jump_backward (-d);
      return;
    }

    assert (d >= 0);
    Diff_ty rem (d);


    while (rem > 0) {
      Diff_ty avail = std::distance (BiDirBase::curr, BiDirBase::localEnd ());
      assert (avail >= 0);

      if (rem > avail) {
        rem -= avail;

        assert (BiDirBase::row < (BiDirBase::workList->numRows () - 1));
        BiDirBase::nextRow ();

      } else {
        BiDirBase::curr += rem;
        rem = 0;
      }

      BiDirBase::seekValidBegin ();
    }
  }

  void jump_backward (const Diff_ty d) {
    if (d < 0) {
      jump_forward (-d);
      return;
    }

    assert (d >= 0);
    Diff_ty rem (d);

    while (rem > 0) {
      Diff_ty avail = std::distance (BiDirBase::localBegin (), BiDirBase::curr);
      assert (avail >= 0);

      if (rem > avail) {
        rem -= avail;

        assert (BiDirBase::row > 0);
        BiDirBase::prevRow ();

      } else {
        BiDirBase::curr -= rem;
        rem = 0;
      }
    }
  }

  Diff_ty compute_dist (const TwoLevelRandIter& that) const {
    assert (this->workList == that.workList);

    if (this->row > that.row) {
      return -(that.compute_dist (*this));

    } else if (this->row == that.row) {
      return std::distance (this->curr, that.curr);

    } else {
      TwoLevelRandIter tmp (*this);

      Diff_ty d = std::distance (tmp.curr, tmp.curr); // 0

      while (tmp.row < that.row) {
        d += std::distance (tmp.curr, tmp.localEnd ());
        tmp.nextRow ();
      }

      assert (tmp.row == that.row);

      if (tmp.row < tmp.workList->numRows ()) {
        d += std::distance (tmp.curr, that.curr);
      }

      assert (d >= 0);

      return d;
    }
  }


public:

  TwoLevelRandIter (): BiDirBase () {}

  TwoLevelRandIter (PerThrdWL& wl, const GlobalPos& pos): BiDirBase (wl, pos) {}


  TwoLevelRandIter& operator += (Diff_ty d) {
    jump_forward (d);
    return *this;
  }

  TwoLevelRandIter& operator -= (Diff_ty d) {
    jump_backward (d);
    return *this;
  }

  friend TwoLevelRandIter operator + (const TwoLevelRandIter& it, Diff_ty d) {
    TwoLevelRandIter tmp (it);
    tmp += d;
    return tmp;
  }

  friend TwoLevelRandIter operator + (Diff_ty d, const TwoLevelRandIter& it) {
    return (it + d);
  }

  friend TwoLevelRandIter operator - (const TwoLevelRandIter& it, Diff_ty d) {
    TwoLevelRandIter tmp (it);
    tmp -= d;
    return tmp;
  }

  friend Diff_ty operator - (const TwoLevelRandIter& left, const TwoLevelRandIter& right) {

    return right.compute_dist (left);
  }

  typename BiDirBase::Traits::reference operator [] (Diff_ty d) {
    return *((*this) + d);
  }

  friend bool operator < (const TwoLevelRandIter& left, const TwoLevelRandIter& right) {
    assert (left.workList == right.workList);

    return ((left.row == right.row) ? (left.curr < right.curr) : (left.row < right.row));
  }

  friend bool operator <= (const TwoLevelRandIter& left, const TwoLevelRandIter& right) {
    return (left < right) || (left == right);
  }

  friend bool operator > (const TwoLevelRandIter& left, const TwoLevelRandIter& right) {
    return !(left <= right);
  }

  friend bool operator >= (const TwoLevelRandIter& left, const TwoLevelRandIter& right) {
    return !(left < right);
  }


};

template <typename PerThrdWL, typename Iter, typename Cat>
struct ByCat {};

template <typename PerThrdWL, typename Iter> 
struct ByCat<PerThrdWL, Iter, std::forward_iterator_tag> {
  typedef TwoLevelFwdIter<PerThrdWL, Iter> type;
};

template <typename PerThrdWL, typename Iter> 
struct ByCat<PerThrdWL, Iter, std::bidirectional_iterator_tag> {
  typedef TwoLevelBiDirIter<PerThrdWL, Iter> type;
};

template <typename PerThrdWL, typename Iter> 
struct ByCat<PerThrdWL, Iter, std::random_access_iterator_tag> {
  typedef TwoLevelRandIter<PerThrdWL, Iter> type;
};

template <typename PerThrdWL, typename Iter>
struct ChooseIter {

  typedef typename ByCat<PerThrdWL, Iter, typename std::iterator_traits<Iter>::iterator_category>::type type;

};


template <typename PerThrdWL, typename Iter>
typename ChooseIter<PerThrdWL, Iter>::type make_begin (PerThrdWL& wl, Iter dummy) {

  typedef typename ChooseIter<PerThrdWL, Iter>::type Ret_ty;
  return Ret_ty (wl, GLOBAL_BEGIN);
}

template <typename PerThrdWL, typename Iter>
typename ChooseIter<PerThrdWL, Iter>::type make_end (PerThrdWL& wl, Iter dummy) {

  typedef typename ChooseIter<PerThrdWL, Iter>::type Ret_ty;
  return Ret_ty (wl, GLOBAL_END);
}

#endif

} // end namespace HIDDEN


template <typename Cont_tp> 
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

  typedef PerThreadWorkList This_ty;

  // TODO: choose one at the end
#ifndef USE_CUSTOM_TWO_LEVEL_ITER

  typedef HIDDEN::OuterPerThreadWLIter<This_ty> OuterIter;

  typedef typename ChooseTwoLevelIterator<OuterIter, typename Cont_ty::iterator>::type global_iterator;
  typedef typename ChooseTwoLevelIterator<OuterIter, typename Cont_ty::const_iterator>::type global_const_iterator;
  typedef typename ChooseTwoLevelIterator<OuterIter, typename Cont_ty::reverse_iterator>::type global_reverse_iterator;
  typedef typename ChooseTwoLevelIterator<OuterIter, typename Cont_ty::const_reverse_iterator>::type global_const_reverse_iterator;

#else

  typedef typename HIDDEN::ChooseIter<This_ty, typename Cont_ty::iterator>::type global_iterator;
  typedef typename HIDDEN::ChooseIter<This_ty, typename Cont_ty::const_iterator>::type global_const_iterator;
  typedef typename HIDDEN::ChooseIter<This_ty, typename Cont_ty::reverse_iterator>::type global_reverse_iterator;
  typedef typename HIDDEN::ChooseIter<This_ty, typename Cont_ty::const_reverse_iterator>::type global_const_reverse_iterator;

#endif

private:
  typedef Galois::Runtime::PerThreadStorage<Cont_ty*> PerThrdCont_ty;
  PerThrdCont_ty perThrdCont;

  void destroy () {
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      delete *perThrdCont.getRemote (i);
      *perThrdCont.getRemote (i) = NULL;
    }
  }

protected:
  PerThreadWorkList (): perThrdCont () {
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      *perThrdCont.getRemote (i) = NULL;
    }
  }

  void init (const Cont_ty& cont) {
    for (unsigned i = 0; i < perThrdCont.size (); ++i) {
      *perThrdCont.getRemote (i) = new Cont_ty (cont);
    }
  }


  ~PerThreadWorkList () { 
    destroy ();
  }


public:
  unsigned numRows () const { return perThrdCont.size (); }

  Cont_ty& get () { return **(perThrdCont.getLocal ()); }

  const Cont_ty& get () const { return **(perThrdCont.getLocal ()); }

  Cont_ty& get (unsigned i) { return **(perThrdCont.getRemote (i)); }

  const Cont_ty& get (unsigned i) const { return **(perThrdCont.getRemote (i)); }

  Cont_ty& operator [] (unsigned i) { return get (i); }

  const Cont_ty& operator [] (unsigned i) const { return get (i); }


  // TODO: choose one at the end
#ifndef USE_CUSTOM_TWO_LEVEL_ITER

  global_iterator begin_all () { 
    return make_two_level_begin (
        HIDDEN::make_outer_begin (*this), HIDDEN::make_outer_end (*this),  
        local_iterator ()); 
  }

  global_iterator end_all () { 
    return make_two_level_end (
        HIDDEN::make_outer_begin (*this), HIDDEN::make_outer_end (*this),  
        local_iterator ()); 
  }


  global_const_iterator begin_all () const { 
    return make_two_level_begin (
        HIDDEN::make_outer_begin (*this), HIDDEN::make_outer_end (*this),  
        local_const_iterator ()); 
  }

  global_const_iterator end_all () const { 
    return make_two_level_end (
        HIDDEN::make_outer_begin (*this), HIDDEN::make_outer_end (*this),  
        local_const_iterator ()); 
  }

  global_reverse_iterator rbegin_all () { 
    return make_two_level_begin (
        HIDDEN::make_outer_rbegin (*this), HIDDEN::make_outer_rend (*this),  
        local_reverse_iterator ()); 
  }

  global_reverse_iterator rend_all () { 
    return make_two_level_end (
        HIDDEN::make_outer_rbegin (*this), HIDDEN::make_outer_rend (*this),  
        local_reverse_iterator ()); 
  }


  global_const_reverse_iterator rbegin_all () const { 
    return make_two_level_begin (
        HIDDEN::make_outer_rbegin (*this), HIDDEN::make_outer_rend (*this),  
        local_const_reverse_iterator ()); 
  }

  global_const_reverse_iterator rend_all () const { 
    return make_two_level_end (
        HIDDEN::make_outer_rbegin (*this), HIDDEN::make_outer_rend (*this),  
        local_const_reverse_iterator ()); 
  }


#else 

  global_iterator begin_all () { 
    return HIDDEN::make_begin (*this, local_iterator ()); 
  }

  global_iterator end_all () { 
    return HIDDEN::make_end (*this, local_iterator ()); 
  }

  global_const_iterator begin_all () const { 
    return HIDDEN::make_begin (*this, local_const_iterator ()); 
  }
  
  global_const_iterator end_all () const { 
    return HIDDEN::make_end (*this, local_const_iterator ()); 
  }

  global_reverse_iterator rbegin_all () { 
    return HIDDEN::make_begin (*this, local_reverse_iterator ()); 
  }
  
  global_reverse_iterator rend_all () { 
    return HIDDEN::make_end (*this, local_reverse_iterator ()); 
  }

  global_const_reverse_iterator rbegin_all () const { 
    return HIDDEN::make_begin (*this, local_const_reverse_iterator ()); 
  }

  global_const_reverse_iterator rend_all () const { 
    return HIDDEN::make_end (*this, local_const_reverse_iterator ()); 
  }

#endif

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

namespace M = Galois::Runtime::MM;

// TODO: rename to per thread heap factory, move outside
template <typename T>
struct PerThreadAllocatorFactory {


  typedef M::SimpleBumpPtrWithMallocFallback<M::FreeListHeap<M::SystemBaseAlloc> > BasicHeap;

  typedef M::ThreadAwarePrivateHeap<BasicHeap> PerThreadHeap;

  typedef M::ExternRefGaloisAllocator<T, PerThreadHeap> PerThreadAllocator;

};

// TODO: remove code reuse here.
template <typename T>
class PerThreadVector: 
  public PerThreadWorkList<std::vector<T, typename PerThreadAllocatorFactory<T>::PerThreadAllocator> > {

public:
  typedef typename PerThreadAllocatorFactory<T>::PerThreadHeap Heap_ty;
  typedef typename PerThreadAllocatorFactory<T>::PerThreadAllocator Alloc_ty;
  typedef std::vector<T, Alloc_ty> Cont_ty;

protected:
  typedef PerThreadWorkList<Cont_ty> Super_ty;

  Heap_ty heap;
  Alloc_ty alloc;

public:
  PerThreadVector (): Super_ty (), heap (), alloc (&heap) {

    Super_ty::init (Cont_ty (alloc));
  }

  

};


template <typename T>
class PerThreadDeque: 
  public PerThreadWorkList<std::deque<T, typename PerThreadAllocatorFactory<T>::PerThreadAllocator> > {

public:
  typedef typename PerThreadAllocatorFactory<T>::PerThreadHeap Heap_ty;
  typedef typename PerThreadAllocatorFactory<T>::PerThreadAllocator Alloc_ty;
  typedef std::deque<T, Alloc_ty> Cont_ty;

protected:
  typedef PerThreadWorkList<Cont_ty> Super_ty;

  Heap_ty heap;
  Alloc_ty alloc;

public:
  PerThreadDeque (): Super_ty (), heap (), alloc (&heap) {

    Super_ty::init (Cont_ty (alloc));

  }

};

template <typename T, typename C=std::less<T> >
class PerThreadSet: 
  public PerThreadWorkList<std::set<T, C, typename PerThreadAllocatorFactory<T>::PerThreadAllocator> > {

public:
  typedef typename PerThreadAllocatorFactory<T>::PerThreadHeap Heap_ty;
  typedef typename PerThreadAllocatorFactory<T>::PerThreadAllocator Alloc_ty;
  typedef std::set<T, C, Alloc_ty> Cont_ty;

protected:
  typedef PerThreadWorkList<Cont_ty> Super_ty;

  Heap_ty heap;
  Alloc_ty alloc;

public:
  explicit PerThreadSet (const C& cmp=C ()): Super_ty (), heap (), alloc (&heap) {

    Super_ty::init (Cont_ty (cmp, alloc));
  }

};

}
}




#endif // GALOIS_RUNTIME_PER_THREAD_WORK_LIST_H_
