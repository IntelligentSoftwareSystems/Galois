#ifndef FP_WRAPPER_H
#define FP_WRAPPER_H

#include <boost/rational.hpp>

class RationalWrapper : public boost::less_than_comparable<RationalWrapper>,
                        public boost::equality_comparable<RationalWrapper>,
                        public boost::addable<RationalWrapper>,
                        public boost::subtractable<RationalWrapper>,
                        public boost::multipliable<RationalWrapper>,
                        public boost::dividable<RationalWrapper> {

  using Impl = boost::rational<int64_t>;

  static const int64_t MAX_DENOMINATOR = int64_t(1) << 31;
  static const int64_t MAX_NUMERATOR   = int64_t(1) << 31;

  Impl m_val;

  void check(void) const {
    assert(m_val.denominator() >= 0);
    assert(std::abs(m_val.numerator()) <= MAX_NUMERATOR);
    assert(m_val.denominator() <= MAX_DENOMINATOR);
  }

  void truncate(void) {

    assert(double(*this) < MAX_NUMERATOR);

    if (std::abs(m_val.numerator()) >= MAX_NUMERATOR ||
        m_val.denominator() >= MAX_DENOMINATOR) {

      int64_t n = m_val.numerator();
      int64_t d = m_val.denominator();
      assert(d >= 0);

      while (std::abs(n) > MAX_NUMERATOR || d > MAX_DENOMINATOR) {
        n = (n / 2) + (n % 2);
        d = (d / 2) + (d % 2);

        assert(std::abs(n) >= 1);
        assert(d >= 2);
      }

      m_val.assign(n, d);
    }
  }

public:
  RationalWrapper(void) : m_val(0, 1) {}

  RationalWrapper(int64_t n, int64_t d) : m_val(n, d) {}

  template <typename I>
  RationalWrapper(I x) : m_val(x, 1) {
    static_assert(std::is_integral<I>::value, "argument type must be integer");
    // if (std::abs (x) > MAX_NUMERATOR) {
    // std::abort ();
    // }
  }

  RationalWrapper(double d)
      : m_val(int64_t(d * MAX_DENOMINATOR), MAX_DENOMINATOR) {

    if (std::fabs(d) > double(MAX_NUMERATOR)) {
      std::abort();
    }

    this->truncate();
    this->check();
  }

  // operator int64_t (void) const {
  // return boost::rational_cast<int64_t> (m_val);
  // }

  operator double(void) const { return boost::rational_cast<double>(m_val); }

  double dval(void) const { return boost::rational_cast<double>(m_val); }

  std::string str(void) const {
    char s[256];

    std::sprintf(s, "%ld/%ld", m_val.numerator(), m_val.denominator());

    return s;
  }

  friend std::ostream& operator<<(std::ostream& o, const RationalWrapper& r) {
    return (o << r.str());
  }

  // unary - and plus
  const RationalWrapper& operator+(void) const { return *this; }

  RationalWrapper operator-(void) const {
    return RationalWrapper(-(m_val.numerator()), m_val.denominator());
  }

  RationalWrapper& operator+=(const RationalWrapper& that) {

    this->check();
    that.check();

    m_val += that.m_val;

    this->truncate();
    this->check();

    return *this;
  }

  RationalWrapper& operator-=(const RationalWrapper& that) {

    this->check();
    that.check();

    m_val -= that.m_val;

    this->truncate();
    this->check();

    return *this;
  }

  RationalWrapper& operator*=(const RationalWrapper& that) {
    this->check();
    that.check();

    m_val *= that.m_val;

    this->truncate();
    this->check();

    return *this;
  }

  RationalWrapper& operator/=(const RationalWrapper& that) {
    this->check();
    that.check();

    m_val /= that.m_val;

    this->truncate();
    this->check();

    return *this;
  }

  bool operator<(const RationalWrapper& that) const {
    return m_val < that.m_val;
  }

  bool operator==(const RationalWrapper& that) const {
    return m_val == that.m_val;
  }

  static RationalWrapper fabs(const RationalWrapper& r) {
    r.check();
    return RationalWrapper(std::abs(r.m_val.numerator()),
                           r.m_val.denominator());
  }

  static RationalWrapper sqrt(const RationalWrapper& r) {
    double d = double(r);
    assert(d >= 0);
    double ret = std::sqrt(d);

    return RationalWrapper(ret);
  }
};

namespace std {

template <>
class numeric_limits<RationalWrapper> : public std::numeric_limits<int64_t> {};

} // end namespace std

class DoubleWrapper : public boost::less_than_comparable<DoubleWrapper>,
                      public boost::equality_comparable<DoubleWrapper>,
                      public boost::addable<DoubleWrapper>,
                      public boost::subtractable<DoubleWrapper>,
                      public boost::multipliable<DoubleWrapper>,
                      public boost::dividable<DoubleWrapper> {

  using Impl = double;

  Impl m_val;

  static const int64_t SCALING_FACTOR = int64_t(1) << 32;

  static const bool DO_ROUND = false;

  void check(void) const {
    // TODO: uncomment this check after fixing discr code in Collision.h (136)
    // assert (std::fabs (m_val) < double (SCALING_FACTOR));
  }

  void truncate(void) {
    // check ();
    //
    // if (DO_ROUND) {
    // std::abort (); // TODO: implement
    //
    // } else {
    // int64_t x = int64_t (m_val * SCALING_FACTOR);
    // double d = double (x) / double (SCALING_FACTOR);
    // }
  }

public:
  DoubleWrapper(void) : m_val(0.0) {}

  template <typename I>
  DoubleWrapper(I x) : m_val(x) {
    static_assert(std::is_integral<I>::value, "argument type must be integer");
    this->check();
    // if (std::abs (x) > MAX_NUMERATOR) {
    // std::abort ();
    // }
  }

  DoubleWrapper(double d) : m_val(d) { this->truncate(); }

  operator double(void) const { return m_val; }

  double dval(void) const { return m_val; }

  std::string str(void) const {
    char s[256];

    std::sprintf(s, "%10.10lf", m_val);

    return s;
  }

  friend std::ostream& operator<<(std::ostream& o, const DoubleWrapper& r) {
    return (o << r.str());
  }

  // unary - and plus
  const DoubleWrapper& operator+(void) const { return *this; }

  DoubleWrapper operator-(void) const { return DoubleWrapper(-m_val); }

  DoubleWrapper& operator+=(const DoubleWrapper& that) {

    this->check();
    that.check();

    m_val += that.m_val;

    this->truncate();
    this->check();

    return *this;
  }

  DoubleWrapper& operator-=(const DoubleWrapper& that) {

    this->check();
    that.check();

    m_val -= that.m_val;

    this->truncate();
    this->check();

    return *this;
  }

  DoubleWrapper& operator*=(const DoubleWrapper& that) {
    this->check();
    that.check();

    m_val *= that.m_val;

    this->truncate();
    this->check();

    return *this;
  }

  DoubleWrapper& operator/=(const DoubleWrapper& that) {
    this->check();
    that.check();

    m_val /= that.m_val;

    this->truncate();
    this->check();

    return *this;
  }

  bool operator<(const DoubleWrapper& that) const {
    // this->check ();
    // that.check ();
    return m_val < that.m_val;
  }

  bool operator==(const DoubleWrapper& that) const {
    // this->check ();
    // that.check ();
    return m_val == that.m_val;
  }

  static DoubleWrapper fabs(const DoubleWrapper& r) {
    r.check();
    return DoubleWrapper(std::fabs(r.m_val));
  }

  static DoubleWrapper sqrt(const DoubleWrapper& r) {
    r.check();
    double d = double(r);
    assert(d >= 0);
    double ret = std::sqrt(d);

    return DoubleWrapper(ret);
  }
};

namespace std {
template <>
class numeric_limits<DoubleWrapper> : public std::numeric_limits<double> {};
} // namespace std

#endif // FP_WRAPPER_H
