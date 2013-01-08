#ifndef GALOIS_GPTR_H
#define GALOIS_GPTR_H

#if 0
template <class T>
class gptr {
  explicit auto_ptr(T* p = 0) throw();

};
#endif

template<class T>
class gptr {
  T*  ptr;
  int owner;
public:
  typedef T element_type;
  
  gptr() {
    owner = networkHostID;
    ptr = new T;
  }
  constexpr gptr(std::nullptr_t) :ptr(nullptr) {}

  gptr( const gptr& r ) :ptr(r.ptr) {}

  ~gptr() {}

  gptr& operator=(const gptr& sp) {
    ptr = sp.ptr;
    return *this;
  }

  T& operator*() const { return *ptr; }
  T *operator->() const { return ptr; }
  operator bool() const { return ptr != nullptr; }
};

#endif
