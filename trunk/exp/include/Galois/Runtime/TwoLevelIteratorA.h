/** Two Level Iterator for Per-thread workList-*- C++ -*-
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
 * Two Level Iterator for Per-thread workList
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_TWOLEVELITERA_H
#define GALOIS_RUNTIME_TWOLEVELITERA_H

#include <iterator>

#include <cstdlib>

namespace GaloisRuntime {

namespace intern {

enum GlobalPos {
  GLOBAL_BEGIN, GLOBAL_END
};


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
struct IsReverse {
  static const bool VAL = false;
};

template <typename PerThrdWL>
struct IsReverse<PerThrdWL, typename PerThrdWL::Cont_ty::reverse_iterator> {
  static const bool VAL = true;
};

template <typename PerThrdWL>
struct IsReverse<PerThrdWL, typename PerThrdWL::Cont_ty::const_reverse_iterator> {
  static const bool VAL = true;
};

template <typename PerThrdWL, typename Iter>
class TwoLevelFwdIter: 
  public std::iterator_traits<Iter>, 
  public TwoLevelIterBase<PerThrdWL, Iter, IsReverse<PerThrdWL, Iter>::VAL> {

protected:

  typedef std::iterator_traits<Iter> Traits;
  typedef TwoLevelIterBase<PerThrdWL, Iter, IsReverse<PerThrdWL, Iter>::VAL> Base;

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
struct ByCategory {};

template <typename PerThrdWL, typename Iter> 
struct ByCategory<PerThrdWL, Iter, std::forward_iterator_tag> {
  typedef TwoLevelFwdIter<PerThrdWL, Iter> type;
};

template <typename PerThrdWL, typename Iter> 
struct ByCategory<PerThrdWL, Iter, std::bidirectional_iterator_tag> {
  typedef TwoLevelBiDirIter<PerThrdWL, Iter> type;
};

template <typename PerThrdWL, typename Iter> 
struct ByCategory<PerThrdWL, Iter, std::random_access_iterator_tag> {
  typedef TwoLevelRandIter<PerThrdWL, Iter> type;
};

template <typename PerThrdWL, typename Iter>
struct ChooseIter {

  typedef typename ByCategory<PerThrdWL, Iter, typename std::iterator_traits<Iter>::iterator_category>::type type;

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


} // end namespace intern
} // end namespace GaloisRuntime

#endif // GALOIS_RUNTIME_TWO_LEVEL_ITER_A_H

