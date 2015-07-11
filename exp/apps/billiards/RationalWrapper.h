#ifndef RATIONAL_WRAPPER_H_
#define RATIONAL_WRAPPER_H_

#include <boost/rational.hpp>

class RationalWrapper: 
  public boost::less_than_comparable<RationalWrapper>,
  public boost::equality_comparable<RationalWrapper>,
  public boost::addable<RationalWrapper>,
  public boost::subtractable<RationalWrapper>,
  public boost::multipliable<RationalWrapper>,
  public boost::dividable<RationalWrapper> {
  
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

  RationalWrapper (void): val (0, 1) {}

  RationalWrapper (int64_t n, int64_t d): val (n, d) {}

  template <typename I>
  RationalWrapper (I x): val (x, 1) {
    static_assert (std::is_integral<I>::value, "argument type must be integer");
    // if (std::abs (x) > MAX_NUMERATOR) {
      // std::abort ();
    // }
  }

  RationalWrapper (double d): val (int64_t (d * MAX_DENOMINATOR), MAX_DENOMINATOR) {

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

  friend std::ostream& operator << (std::ostream& o, const RationalWrapper& r) {
    return (o << r.str ());
  }

  // unary - and plus
  const RationalWrapper& operator + (void) const {
    return *this;
  }

  RationalWrapper operator - (void) const {
    return RationalWrapper (-(val.numerator ()), val.denominator ());
  }

  RationalWrapper& operator += (const RationalWrapper& that) {

    this->check ();
    that.check ();

    val += that.val;

    this->truncate ();
    this->check ();

    return *this;
  }

  RationalWrapper& operator -= (const RationalWrapper& that) {

    this->check ();
    that.check ();

    val -= that.val;

    this->truncate ();
    this->check ();

    return *this;
  }

  RationalWrapper& operator *= (const RationalWrapper& that) {
    this->check ();
    that.check ();

    val *= that.val;

    this->truncate ();
    this->check ();

    return *this;
  }

  RationalWrapper& operator /= (const RationalWrapper& that) {
    this->check ();
    that.check ();

    val /= that.val;

    this->truncate ();
    this->check ();

    return *this;
  }

  bool operator < (const RationalWrapper& that) const {
    return val < that.val;
  }

  bool operator == (const RationalWrapper& that) const { 
    return val == that.val;
  }

  static RationalWrapper fabs (const RationalWrapper& r) {
    r.check ();
    return RationalWrapper (std::abs (r.val.numerator ()), r.val.denominator ());
  }

  static RationalWrapper sqrt (const RationalWrapper& r) {
    double d = double (r);
    assert (d >= 0);
    double ret = std::sqrt (d);

    return RationalWrapper (ret);
  }
};


namespace std {

  template <>
  class numeric_limits<RationalWrapper>: public std::numeric_limits<int64_t> {};

} // end namespace std

#endif // RATIONAL_WRAPPER_H_
