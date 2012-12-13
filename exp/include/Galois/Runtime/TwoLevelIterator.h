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

#ifndef GALOIS_RUNTIME_TWO_LEVEL_ITER_H
#define GALOIS_RUNTIME_TWO_LEVEL_ITER_H

#include <iterator>

#include <cstdlib>
#include <cassert>

namespace Galois {
namespace Runtime {

namespace intern {

//XXX: Important pitfalls to handles
// 1 - If Outer and Inner have different categories, which category to choose?. If
// Outer is at least bidirectional, then the category of Inner can be chosen, else,
// the lowest category of Outer and Inner can be chosen
//
// 2 - Prevent Outer from falling outside the [begin,end) range, because calling
// container functions e.g. outer->begin () and outer->end () is not valid and may
// cause a segfault
//
// 3 - The initial position of Outer and Inner iterators must be such that calling
// operator * or operator -> on at two level iterator yields a valid result (if
// possible). This means advancing the inner iterator to begin of first non-empty
// container (else to the end of outer). If the outer iterator is initialized to
// end of the outer range i.e. [end, end), then inner iterator cannot be
// initialized
//
// 4 - When incrementing (++), the inner iterator should initially be at a valid
// begin position, but after incrementing may end up at end of an Inner iterator.
// So the next valid local begin must be found, else the end of 
// outer should be reached
//
// 4.1 - When jumping forward, outer should not go beyond end. After jump is
// completed, inner may be at local end, so a valid next begin must be found or
// else end of outer must be reached
//
// 5 - When decrementing (--), the inner iterator may initially be uninitialized
// due to outer being at end (See 3 above)
// Inner iterator must be brought to a valid location after decrementing, or, else
// the begin of outer must be reached (and not exceeded). 
//
// 5.1- When jumping backward, inner iterator may be uninitialized due to outer
// being at end.  
//
// 6 - When jumping forward or backward, check for jump amount being negavite.  
// 6.1 - Jumping outside the range of outer cannot be supported. 

template <typename Outer, typename Inner, typename Cat, bool is_reverse_tp>
class TwoLevelIterBase {

protected:
  // TODO: make begin and end const
  Outer m_beg_outer;
  Outer m_end_outer;
  Outer m_outer;
  Inner m_inner;

  inline Inner innerBegin () const { 
    return m_outer->begin ();
  }

  inline Inner innerEnd () const {
    return m_outer->end  ();
  }

  TwoLevelIterBase (): m_outer (), m_end_outer (), m_inner () {}

  TwoLevelIterBase (Outer beg_outer, Outer end_outer)
    : m_beg_outer (beg_outer), 
      m_end_outer (end_outer), 
      m_outer (m_beg_outer),
      m_inner () 
  {}

};

template <typename Outer, typename Inner, typename Cat>
class TwoLevelIterBase<Outer, Inner, Cat, true> {

protected:
  // TODO: make begin and end const
  Outer m_beg_outer;
  Outer m_end_outer;
  Outer m_outer;
  Inner m_inner;

  inline Inner innerBegin () const { 
    return m_outer->rbegin ();
  }

  inline Inner innerEnd () const {
    return m_outer->rend  ();
  }

  TwoLevelIterBase (): m_outer (), m_end_outer (), m_inner () {}

  TwoLevelIterBase (Outer beg_outer, Outer end_outer)
    : m_beg_outer (beg_outer), 
      m_end_outer (end_outer), 
      m_outer (m_beg_outer),
      m_inner () 
  {}

};

// TODO: move into hidden namespace
template <typename Iter, typename Cat>
struct OverrideCategory: 
  public std::iterator<
    Cat,
    typename std::iterator_traits<Iter>::value_type,
    typename std::iterator_traits<Iter>::difference_type,
    typename std::iterator_traits<Iter>::pointer,
    typename std::iterator_traits<Iter>::reference
  > 
{};

template <typename Outer, typename Inner, typename Cat, bool is_reverse_tp>
class TwoLevelFwdIter: 
  public OverrideCategory<Inner, Cat>,
  public TwoLevelIterBase<Outer, Inner, Cat, is_reverse_tp> {

protected:

  typedef OverrideCategory<Inner, Cat> Traits;
  typedef TwoLevelIterBase<Outer, Inner, Cat, is_reverse_tp> Base;

  inline bool innerAtBegin () const {
    return Base::m_inner == Base::innerBegin ();
  }

  inline bool innerAtEnd () const {
    return Base::m_inner == Base::innerEnd ();
  }

  inline bool outerAtBegin () const {
    return Base::m_outer == Base::m_beg_outer;
  }

  inline bool outerAtEnd () const {
    return Base::m_outer == Base::m_end_outer;
  }

  inline bool outerEmpty () const {
    return Base::m_beg_outer == Base::m_end_outer;
  }


  void nextInner () {
    assert (!outerAtEnd ());
    ++Base::m_outer;
    if (!outerAtEnd ()) {
      Base::m_inner = Base::innerBegin ();
    }
  }

  void seekValidBegin () {
    while (!outerAtEnd () && innerAtEnd ()) {
      nextInner ();
    } 
  }


  void step_forward () {
    assert (!innerAtEnd ());
    ++Base::m_inner;

    if (innerAtEnd ()) {
      seekValidBegin ();
    }
  }

  bool is_equal (const TwoLevelFwdIter& that) const {
    assert (this->m_beg_outer == that.m_beg_outer);
    assert (this->m_end_outer == that.m_end_outer);

    return (this->m_outer == that.m_outer) 
      && (outerAtEnd () || (this->m_inner == that.m_inner));
  }


public:

  TwoLevelFwdIter (): Base () {}

  TwoLevelFwdIter (Outer beg_outer, Outer end_outer)
    : Base (beg_outer, end_outer) {

    if (!outerAtEnd ()) {
      Base::m_inner = Base::innerBegin ();
      seekValidBegin ();
    }
  }

  typename Traits::reference operator * () { 
    return *Base::m_inner;
  }

  typename Traits::pointer operator -> () {
    return Base::m_inner->operator -> ();
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

template <typename Outer, typename Inner, typename Cat, bool is_reverse_tp>
class TwoLevelBiDirIter: public TwoLevelFwdIter<Outer, Inner, Cat, is_reverse_tp> {

protected:
  typedef TwoLevelFwdIter<Outer, Inner, Cat, is_reverse_tp> FwdBase;

  void prevInner () {
    assert (!FwdBase::outerAtBegin ());
    --FwdBase::m_outer;
    FwdBase::m_inner = FwdBase::innerEnd ();
  }

  void step_backward () {

    assert (!FwdBase::outerEmpty ());

    // calling innerBegin when m_outer == m_end_outer is invalid
    // so call prevInner first
    while (!FwdBase::outerAtBegin ()) {
      FwdBase::prevInner ();
      if (!FwdBase::innerAtBegin ()) {
        break;
      }
    }

    if (FwdBase::innerAtBegin ()) {
      assert (FwdBase::outerAtBegin ());

    } else {
      --FwdBase::m_inner;
    }
  }

public:

  TwoLevelBiDirIter (): FwdBase () {}

  TwoLevelBiDirIter (Outer beg_outer, Outer end_outer)
    : FwdBase (beg_outer, end_outer) {}


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


template <typename Outer, typename Inner, typename Cat, bool is_reverse_tp>
class TwoLevelRandIter: public TwoLevelBiDirIter<Outer, Inner, Cat, is_reverse_tp> {

protected:
  typedef TwoLevelBiDirIter<Outer, Inner, Cat, is_reverse_tp> BiDirBase;

  typedef typename BiDirBase::Traits::difference_type Diff_ty;

  void jump_forward (const Diff_ty d) {
    assert (!BiDirBase::outerEmpty ());

    if (d < 0) {
      jump_backward (-d);

    } else {
      Diff_ty rem (d);

      while (rem > 0) {
        assert (!outerAtEnd ());

        Diff_ty avail = std::distance (BiDirBase::m_inner, BiDirBase::innerEnd ());
        assert (avail >= 0);

        if (rem > avail) {
          rem -= avail;
          assert (!outerAtEnd ());
          BiDirBase::nextInner ();

        } else {
          BiDirBase::curr += rem;
          rem = 0;
        }

        BiDirBase::seekValidBegin ();
      }
    }
  }

  void jump_backward (const Diff_ty d) {
    assert (!BiDirBase::outerEmpty ());

    if (d < 0) {
      jump_forward (-d);

    } else {


      Diff_ty rem (d);

      if ((rem > 0) && BiDirBase::outerAtEnd ()) {
        BiDirBase::prevInner ();
        
      }

      while (rem > 0) {
        Diff_ty avail = std::distance (BiDirBase::innerBegin (), BiDirBase::m_inner);
        assert (avail >= 0);

        if (rem > avail) {
          rem -= avail;
          assert (!BiDirBase::outerAtBegin ());
          BiDirBase::prevInner ();
          
        } else {

          BiDirBase::m_inner -= rem;
          rem = 0;
          break;
        }
      }
    }
  }

  Diff_ty compute_dist (const TwoLevelRandIter& that) const {

    if (std::distance (this->m_outer, that.m_outer) < 0) { // this->m_outer > that.m_outer
      return -(that.compute_dist (*this));

    } else if (this->m_outer == that.m_outer) {
      return std::distance (this->m_inner, that.m_inner);

    } else { 

      assert (std::distance (this->m_outer, that.m_outer) > 0); // this->m_outer < that.m_outer;

      TwoLevelRandIter tmp (*this);

      Diff_ty d = tmp.m_inner - tmp.m_inner; // 0

      while (tmp.m_outer != that.m_outer) {
        d += std::distance (tmp.m_inner, tmp.innerEnd ());
        tmp.nextInner ();
      }

      assert (tmp.m_outer == that.m_outer);

      if (tmp.m_outer != tmp.m_end_outer) {
        d += std::distance (tmp.m_inner, that.m_inner);
      }

      assert (d >= 0);

      return d;
    }
  }


public:

  TwoLevelRandIter (): BiDirBase () {}

  TwoLevelRandIter (Outer beg_outer, Outer end_outer)
    : BiDirBase (beg_outer, end_outer) {}

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
    return ((left.m_outer == right.m_outer) ? (left.m_inner < right.m_inner) : (left.m_outer < right.m_outer));
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

template <typename CatOuter, typename CatInner>
struct ChooseCategory {};

// template <typename Cat> 
// struct ChooseCategory<Cat, Cat> {
  // typedef Cat category;
// };
// 
// If outer is random_access_iterator_tag or bidirectional_iterator_tag
// then choose inner
template <typename Cat>
struct ChooseCategory<std::random_access_iterator_tag, Cat> {
  typedef Cat category;
};

template <typename Cat>
struct ChooseCategory<std::bidirectional_iterator_tag, Cat> {
  typedef Cat category;
};

// if outer is forward_iterator_tag, then inner is restricted for
// forward_iterator_tag. XXX: Assuming inner is always forward_iterator_tag or
// better
template <typename Cat>
struct ChooseCategory<std::forward_iterator_tag, Cat> {
  typedef std::forward_iterator_tag category;
};

template <typename Outer, typename Inner, typename Cat, bool is_reverse_tp>
struct ByCategory {};

template <typename Outer, typename Inner, bool is_reverse_tp> 
struct ByCategory<Outer, Inner, std::forward_iterator_tag, is_reverse_tp> {
  typedef TwoLevelFwdIter<Outer, Inner, std::forward_iterator_tag, is_reverse_tp> type;
};

template <typename Outer, typename Inner, bool is_reverse_tp> 
struct ByCategory<Outer, Inner, std::bidirectional_iterator_tag, is_reverse_tp> {
  typedef TwoLevelBiDirIter<Outer, Inner, std::bidirectional_iterator_tag, is_reverse_tp> type;
};

template <typename Outer, typename Inner, bool is_reverse_tp> 
struct ByCategory<Outer, Inner, std::random_access_iterator_tag, is_reverse_tp> {
  typedef TwoLevelRandIter<Outer, Inner, std::random_access_iterator_tag, is_reverse_tp> type;
};


template <typename Outer, typename Inner, bool is_reverse_tp=false>
struct ChooseIterator {
private:

  typedef typename std::iterator_traits<Outer>::iterator_category CatOuter;
  typedef typename std::iterator_traits<Inner>::iterator_category CatInner;

public:
  typedef typename ByCategory<Outer, Inner, typename ChooseCategory<CatOuter, CatInner>::category, is_reverse_tp>::type type;

};


enum GlobalPos {
  // TODO: change to OUTER_BEGIN ...
  GLOBAL_BEGIN, GLOBAL_END
};

template <typename PerThrdWL, typename Cont_tp>
class OuterPerThreadWLIter: public std::iterator<std::random_access_iterator_tag, Cont_tp> {

  typedef std::iterator<std::random_access_iterator_tag, Cont_tp> Super_ty;
  typedef typename Super_ty::difference_type Diff_ty;

  PerThrdWL* workList;
  // using ptrdiff_t due to reverse iterator, whose 
  // end is -1, and,  begin is numRows - 1
  ptrdiff_t row;

  void assertInRange () const {
    assert ((row >= 0) && (row < workList->numRows ()));
  }

  Cont_tp& getWL () {
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

  typename Super_ty::pointer operator -> () { return &(getWL ()); }

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


} // end namespace intern
} // end namespace Galois::Runtime
}

#endif // GALOIS_RUNTIME_TWO_LEVEL_ITER_H
