#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <vector>

template <typename T>
std::ostream& operator << (std::ostream& out, const std::vector<T>& v) {
  out << "{ ";
  for (typename std::vector<T>::const_iterator i = v.begin(); i != v.end(); ++i) {
    out << *i << ", ";
  }
  out << "}";

  return out;
}

template <typename I>
void printIter (std::ostream& out, I begin, I end) {
  out << "{ ";
  for (I i = begin; i != end; ++i) {
    out << *i << ", ";
  }
  out << "}" << std::endl;
}

#endif
