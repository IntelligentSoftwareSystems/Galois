/** FP utils and constants  -*- C++ -*-
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
 * FP utils and constants.
 *
 * @author <ahassaan@ices.utexas.edu>
 */



#ifndef _FP_UTILS_H_
#define _FP_UTILS_H_

#include <string>
#include <iostream>

#include <boost/noncopyable.hpp>
#include <boost/rational.hpp>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstdint>

#include "fixed_point.h"



class MyRational: 
  public boost::less_than_comparable<MyRational>,
  public boost::equality_comparable<MyRational>,
  public boost::addable<MyRational>,
  public boost::subtractable<MyRational>,
  public boost::multipliable<MyRational>,
  public boost::dividable<MyRational> {
  
  using Impl = boost::rational<int64_t>;

  static const int64_t MAX_DENOMINATOR = int64_t (1) << 31; 
  static const int64_t MAX_NUMERATOR = int64_t (1) << 31;

  Impl val;
    

  void check (void) const {
    assert (val.denominator () >= 0);
    assert (std::abs (val.numerator ()) <= MAX_NUMERATOR);
    assert (val.denominator () <= MAX_DENOMINATOR);
  }

  void truncate (void) {

    assert (double (*this) < MAX_NUMERATOR);

    if (std::abs (val.numerator ()) >= MAX_NUMERATOR || val.denominator () >= MAX_DENOMINATOR) {

      int64_t n = val.numerator ();
      int64_t d = val.denominator ();
      assert (d >= 0);

      while (std::abs (n) > MAX_NUMERATOR || d > MAX_DENOMINATOR) {
        n = (n / 2) + (n % 2);
        d = (d / 2) + (d % 2);

        assert (std::abs (n) >= 1);
        assert (d >= 2);
      }

      val.assign (n, d);
    }
  }

public:

  MyRational (void): val (0, 1) {}

  MyRational (int64_t n, int64_t d): val (n, d) {}

  template <typename I>
  MyRational (I x): val (x, 1) {
    static_assert (std::is_integral<I>::value, "argument type must be integer");
    // if (std::abs (x) > MAX_NUMERATOR) {
      // std::abort ();
    // }
  }

  MyRational (double d): val (int64_t (d * MAX_DENOMINATOR), MAX_DENOMINATOR) {

    if (std::fabs (d) > double (MAX_NUMERATOR)) { 
      std::abort (); 
    }

    this->truncate ();
    this->check ();
  }

  operator int64_t (void) const { 
    return boost::rational_cast<int64_t> (val);
  }

  operator double (void) const {
    return boost::rational_cast<double> (val);
  }

  double dval (void) const {
    return boost::rational_cast<double> (val);
  }

  std::string str (void) const { 
    char s[256];

    std::sprintf (s, "%ld/%ld", val.numerator (), val.denominator ());

    return s;
  }

  friend std::ostream& operator << (std::ostream& o, const MyRational& r) {
    return (o << r.str ());
  }

  // unary - and plus
  const MyRational& operator + (void) const {
    return *this;
  }

  MyRational operator - (void) const {
    return MyRational (-(val.numerator ()), val.denominator ());
  }

  MyRational& operator += (const MyRational& that) {

    this->check ();
    that.check ();

    val += that.val;

    this->truncate ();
    this->check ();

    return *this;
  }

  MyRational& operator -= (const MyRational& that) {

    this->check ();
    that.check ();

    val -= that.val;

    this->truncate ();
    this->check ();

    return *this;
  }

  MyRational& operator *= (const MyRational& that) {
    this->check ();
    that.check ();

    val *= that.val;

    this->truncate ();
    this->check ();

    return *this;
  }

  MyRational& operator /= (const MyRational& that) {
    this->check ();
    that.check ();

    val /= that.val;

    this->truncate ();
    this->check ();

    return *this;
  }

  bool operator < (const MyRational& that) const {
    return val < that.val;
  }

  bool operator == (const MyRational& that) const { 
    return val == that.val;
  }

  static MyRational fabs (const MyRational& r) {
    r.check ();
    return MyRational (std::abs (r.val.numerator ()), r.val.denominator ());
  }

  static MyRational sqrt (const MyRational& r) {
    double d = double (r);
    assert (d >= 0);
    double ret = std::sqrt (d);

    return MyRational (ret);
  }
};


namespace std {

  template <>
  class numeric_limits<MyRational>: public std::numeric_limits<int64_t> {};

} // end namespace std


template <typename T>
struct FPutilsGeneric: private boost::noncopyable {

  static const T EPSILON;
  static const T ERROR_LIMIT;

  static T sqrt (const T& t) {
    double d = std::sqrt (double (t));
    T r = T::sqrt (t);
    
    assert (std::fabs ((r.dval () - d) / d) < 1e-5);

    return r;
  }

  static bool almostEqual (const T& d1, const T& d2) {
    return (T::fabs (d1 - d2) < EPSILON);
  }


  template <typename V>
  static bool almostEqual (const V& v1, const V& v2) {
    return almostEqual (v1.getX (), v2.getX ()) && almostEqual (v1.getY (), v2.getY ());
  }


  //! checks relative error
  static bool checkError (const T& original, const T& measured, bool useAssert=true) {

    T err = T::fabs (measured - original);
    if (original != T (0.0)) {
      err = T::fabs ((measured - original) / original);
    }

    bool withinLim = err < ERROR_LIMIT;


    if (!withinLim) {
      fprintf (stderr, "WARNING: FPutils::checkError, relative error=%10.20g above limit, ERROR_LIMIT=%10.20g\n"
          , double (err), double (ERROR_LIMIT));
    }

    if (useAssert) {
      assert ( withinLim && "Relative Error above limit?");
    }


    return withinLim;
  }



  template <typename V>
  static bool checkError (const V& original, const V& measured, bool useAssert=true) {
    return checkError (original.getX (), measured.getX (), useAssert) 
      && checkError (original.getY (), measured.getY (), useAssert);
  }

  static const T& truncate (const T& val) { return val; }

  template <typename V>
  static const V& truncate (const V& vec) { return vec; }

};

template <typename T>
const T FPutilsGeneric<T>::EPSILON = double (1.0 / (1 << 30));

template <typename T>
const T FPutilsGeneric<T>::ERROR_LIMIT = double (1.0 / (1 << 18));



using FP = fpml::fixed_point<int64_t, 31, 32>;
// using FP = MyRational;
using FPutils = FPutilsGeneric<FP>;


#endif // _FP_UTILS_H_
