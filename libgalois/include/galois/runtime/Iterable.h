#ifndef GALOIS_RUNTIME_ITERABLE_H
#define GALOIS_RUNTIME_ITERABLE_H

namespace galois {
namespace runtime {

//iterable and make_iterable specific
//From: https://github.com/CppCon/CppCon2014/tree/master/Presentations/C%2B%2B11%20in%20the%20Wild%20-%20Techniques%20from%20a%20Real%20Codebase
//Author: Arthur O'Dwyer
//License: The C++ code in this directory is placed in the public domain and may be reused or modified for any purpose, commercial or non-commercial.

template<class It>
class iterable
{
  It m_first, m_last;
public:
  iterable() = default;
  iterable(It first, It last) :
    m_first(first), m_last(last) {}
  It begin() const { return m_first; }
  It end() const { return m_last; }
};

template<class It>
static inline iterable<It> make_iterable(It a, It b)
{
  return iterable<It>(a, b);
}

} // end namespace runtime
} // end namespace galois

#endif //GALOIS_RUNTIME_ITERABLE_H
