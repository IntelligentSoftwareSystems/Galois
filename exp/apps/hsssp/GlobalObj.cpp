

#include "GlobalObj.h"

#include <cassert>

std::vector<uintptr_t> GlobalObject::allobjs;

uintptr_t GlobalObject::ptrForObj(unsigned oid) {
  assert(oid < allobjs.size());
  return allobjs[oid];
}

