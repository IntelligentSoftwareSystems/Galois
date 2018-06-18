#ifndef _ITEMGEN_INCLUDED
#define _ITEMGEN_INCLUDED

#include <iostream>
#include <algorithm>
#include "utils.h"

namespace dataGen {

#define HASH_MAX_INT ((unsigned)1 << 31)

template <class T>
T hash(int i);

template <>
int hash<int>(int i) {
  return utils::hash(i) & (HASH_MAX_INT - 1);
}

template <>
unsigned int hash<unsigned int>(int i) {
  return utils::hash(i);
}

template <>
double hash<double>(int i) {
  return ((double)hash<int>(i) / ((double)HASH_MAX_INT));
}

}; // namespace dataGen

#endif // _ITEMGEN_INCLUDED
