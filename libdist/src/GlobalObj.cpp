#include "galois/runtime/GlobalObj.h"

std::vector<uintptr_t> galois::runtime::GlobalObject::allobjs;

uintptr_t galois::runtime::GlobalObject::ptrForObj(unsigned oid) {
  assert(oid < allobjs.size());
  return allobjs[oid];
}
