#include "Galois/Bag.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>
#include <algorithm>
#include <set>

static const unsigned int NUM_ELEMENTS = 2000;

static void check(const char* func, int expected, int actual) {
  if (expected != actual) {
    std::cerr << func 
      << ": Expected " << expected << " got " << actual << "\n";
    abort();
  }
}

template<typename T>
struct Checker {
  typedef typename T::value_type value_type;
  typedef typename T::iterator iterator;
  
  T& m_c;
  Checker(T& c): m_c(c) { }

  void checkUnique(int n) {
    std::set<value_type> s;
    std::copy(m_c.begin(), m_c.end(), std::inserter(s, s.begin()));
    check(__FUNCTION__, n, s.size());
  }

  void checkRandomAccess() {
    const int modulus = 10;
    size_t dist = std::distance(m_c.begin(), m_c.end());
    iterator begin = m_c.begin();

    for (size_t i = 0; i < dist; i += modulus) {
      if ((*begin % modulus) != 0) {
        std::cerr << __FUNCTION__
          << ": Expected element divisible by " << modulus 
          << " got " << *begin << "\n";
      }
      begin += modulus;
    }
  }
};

template<typename Container>
void testSerial(int n) {
  Container c;
  std::copy(boost::counting_iterator<int>(0),
      boost::counting_iterator<int>(n),
      std::back_inserter(c));

  Checker<Container> checker(c);
  
  checker.checkUnique(n);
  checker.checkRandomAccess();

  int start = n / 2;
  typename Container::iterator b = c.begin();
  std::advance(b, start);
  for (int i = start; i < n; ++i, ++b) {
    check(__FUNCTION__, i, *b);
  }

  Container c2;
  std::copy(boost::counting_iterator<int>(0),
      boost::counting_iterator<int>(n),
      std::back_inserter(c2));

  size_t s = c.size();
  c.splice(c2);
  check(__FUNCTION__, 2*s, c.size());

  checker.checkUnique(n);

  check(__FUNCTION__, n - 1, c.back());

  start = c.size() / 2 - 1;
  typename Container::iterator e = c.begin();
  b = e;
  start = n + 1;
  std::advance(b, start);
  std::advance(e, c.size());
  while (b != e) {
    check(__FUNCTION__, *b++, start++ - n);
  }

  for (size_t size = c.size(); size > 0; --size) {
    check(__FUNCTION__, size, c.size());
    c.pop_back();
  }


}


int main() {
  testSerial<Galois::Bag<int> >(NUM_ELEMENTS);
  testSerial<Galois::SmallBag<int, 5> >(NUM_ELEMENTS);
  testSerial<Galois::SmallBag<int, 10> >(10);
}

