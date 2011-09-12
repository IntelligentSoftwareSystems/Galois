// User-visible allocators -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef GALOIS_MEM_H
#define GALOIS_MEM_H

#include "Galois/Runtime/mem.h"

namespace Galois {

typedef GaloisRuntime::MM::SimpleBumpPtr<GaloisRuntime::MM::FreeListHeap<GaloisRuntime::MM::SystemBaseAlloc> > ItAllocBaseTy;

typedef GaloisRuntime::MM::ExternRefGaloisAllocator<char, ItAllocBaseTy> PerIterAllocTy;

template<typename Ty>
class Allocator;

template<>
class Allocator<void> {
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  template<typename Other>
  struct rebind { typedef Allocator<Other> other; };
};

template<typename Ty>
class Allocator {
  inline void destruct(char*) {}
  inline void destruct(wchar_t*) { }
  template<typename T> inline void destruct(T* t) { t->~T(); }

public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef Ty *pointer;
  typedef const Ty *const_pointer;
  typedef Ty& reference;
  typedef const Ty& const_reference;
  typedef Ty value_type;

  template<class Other>
  struct rebind { typedef Allocator<Other> other; };

  Allocator() throw() { }
  Allocator(const Allocator&) throw() { }
  template <class U> Allocator(const Allocator<U>&) throw() { }
  ~Allocator() { }

  pointer address(reference val) const { return &val; }
  const_pointer address(const_reference val) const { return &val; }

  pointer allocate(size_type size) {
    if (size > max_size())
      throw std::bad_alloc();
#ifdef SOLARIS
    return static_cast<Ty*>(malloc(size * sizeof(Ty)));
#else
    return static_cast<Ty*>(malloc(size * sizeof(Ty)));
#endif
  }

  void deallocate(pointer p, size_type) {
    //::operator delete(p);
#ifdef SOLARIS
    free(p);
#else
    free(p);
#endif
  }

  size_type max_size() const throw() { 
    return size_t(-1) / sizeof(Ty);
  }

  void construct(pointer p, const Ty& val) {
    new(p) Ty(val);
  }

  void destroy(pointer p) {
    destruct(p);
  }
};

template<typename T1, typename T2>
inline bool operator==(const Allocator<T1>&, const Allocator<T2>&) throw() {
  return true;
}

template<typename T1, typename T2>
inline bool operator!=(const Allocator<T1>&, const Allocator<T2>&) throw() {
  return false;
}
}

#endif
