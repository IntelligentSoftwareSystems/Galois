#ifndef GALOIS_GPTR_H
#define GALOIS_GPTR_H

template <class T>
class gptr {
  explicit auto_ptr(T* p = 0) throw();

};

template<class T>
class gptr {
  T*  ptr;
  int owner;
public:
  typedef T element_type;
  
  constexpr gptr() :ptr(nullptr) {}
  constexpr gptr(std::nullptr_t) :ptr(nullptr) {}

  gptr( const gptr& r ) :ptr(r.ptr) {}

  ~gptr();

  gptr& operator=(const gptr& sp) {
    ptr = sp.ptr;
    return *this;
  }

  Ty& operator*() const { return *ptr; }
  Ty *operator->() const { return ptr; }
  operator bool() const { return ptr != nullptr; }
};

#endif
