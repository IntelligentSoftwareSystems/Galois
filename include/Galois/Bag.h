/** Bags -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_BAG_H
#define GALOIS_BAG_H

#include "Galois/Runtime/InsBag.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/mm/Mem.h"

#include <iterator>

namespace Galois {

/**
 * Bag for only concurrent insertions. This data structure
 * supports scalable concurrent pushes but reading the bag
 * can only be done serially.
 */
template<typename T>
struct InsertBag: public GaloisRuntime::galois_insert_bag<T> { };

/**
 * Bag for serial use.
 */
// TODO(ddn): POD specialization
// TODO(ddn): Factor out template dependencies to reduce code bloat
template<class T, int blockSize=1024*16>
class Bag: private boost::noncopyable {
  typedef GaloisRuntime::MM::SimpleBumpPtr<GaloisRuntime::MM::FreeListHeap<GaloisRuntime::MM::SystemBaseAlloc> > Allocator;
  Allocator alloc;

protected:  
  struct Header {
    Header* m_next;
    Header* m_prev;
    T* m_begin; //start of interesting data
    T* m_end;   //end of valid data
    T* m_last;  //end of storage
  };

  //! Raw and aligned space for N elements of type T.
  union U {
    double d;
    long double ld;
    long long l;
    void *p;
  };

  Header* m_head;
  size_t m_size;
  U m_first;

  //! Insert header at end of linked list
  void insertBack(Header* h) {
    if (m_head) {
      h->m_next = m_head;
      h->m_prev = m_head->m_prev;
      m_head->m_prev->m_next = h;
      m_head->m_prev = h;
    } else {
      m_head = h;
    }
  }

  static Header* initHeader(Header* h, T* begin, T* last) {
    h->m_begin = h->m_end = begin;
    h->m_last = last;
    h->m_next = h->m_prev = h;
    return h;
  }

  bool isInlined() const {
    return m_head && m_head->m_begin == static_cast<const void*>(&m_first);
  }

  void destroyRange(T* begin, T* end) {
    while (begin != end) {
      begin->~T();
      ++begin;
    }
  }

#if 0
  static const size_t block_size = 128;
  void freeHeader(Header* h) {
    if (h->m_begin == static_cast<void*>(&m_first))
      return;

    memset(h->m_begin, 0, block_size*sizeof(T));
    free(h->m_begin);
    memset(h, 0, sizeof(*h));
    delete(h);
  }

  Header* newHeader() {
    Header* h = new Header();
    T* begin = reinterpret_cast<T*>(malloc(block_size*sizeof(T)));
    T* last = &begin[block_size];
    return initHeader(h, begin, end);
  }
#else
  void freeHeader(Header* h) {
    if (h->m_begin == static_cast<void*>(&m_first))
      return;

    alloc.deallocate(h);
  }

  Header* newHeader() {
    void* m = alloc.allocate(blockSize);
    Header* h = new (m) Header();
    int offset = 1;
    if (sizeof(T) < sizeof(Header))
      offset += sizeof(Header)/sizeof(T);
    T* a = reinterpret_cast<T*>(m);
    return initHeader(h, &a[offset], &a[(blockSize / sizeof(T))]);
  }
#endif

  //! Convert inline representation to out-of-line one
  void explode() {
    assert(isInlined());

    Header* h = newHeader();

    // TODO(ddn): Probably could fix this...
    if (std::distance(m_head->m_begin, m_head->m_end) > std::distance(h->m_begin, h->m_last)) {
      GALOIS_ERROR(true, "more inline elements than will fit in block");
    }
    h->m_end = std::copy(m_head->m_begin, m_head->m_end, h->m_begin);
    destroyRange(m_head->m_begin, m_head->m_end);

    Header *last = m_head->m_prev;
    Header *next = m_head->m_next;
    // Splice new head at front
    if (last != m_head) {
      last->m_next = h;
      h->m_prev = last;
      h->m_next = next;
      next->m_prev = h;
    }

    m_head = h;
  }

public:
  Bag(): m_head(0), m_size(0) { }

  ~Bag() {
    clear();
  }

  typedef T        value_type;
  typedef const T& const_reference;
  typedef T&       reference;
  
  class iterator : public std::iterator<std::random_access_iterator_tag, T> {
    Header* m_head;
    Header* p;
    T* v;
    size_t m_size;

  public:
    iterator (): 
      m_head (NULL),
      p (NULL),
      v (NULL),
      m_size (0)
    {}


    iterator(Header* h, Header* x, T* e, size_t s): m_head(h), p(x), v(e), m_size(s) { }

    // ------ forward iterator concepts --------
    iterator& operator++() {
      ++v;
      ++m_size;
      if (v == p->m_end && p->m_next != m_head) {
        // Reached end but not last header
        p = p->m_next;
        v = p->m_begin;
      }
      
      return *this;
    }

    iterator operator++(int) {
      iterator tmp(*this);
      operator++();
      return tmp;
    }

    bool operator==(const iterator& rhs) const {
      return (p == rhs.p && v == rhs.v);
    }

    bool operator!=(const iterator& rhs) const {
      return !(p == rhs.p && v == rhs.v);
    }

    const_reference operator*() const {
      return *v; 
    }

    reference operator*() {
      return *v;
    }

    // ------ bidirectional iterator concepts --------
    iterator& operator--() {
      if (v == p->m_begin) {
        p = p->m_prev;
        v = p->m_end;
      } else {
        --v;
      }
      --m_size;
      return *this;
    }

    iterator operator--(int) {
      iterator tmp(*this);
      operator--();
      return tmp; 
    }

    // ------- random access iterator concepts -------
    typedef ptrdiff_t difference_type;

    iterator& operator+=(difference_type n) {
      while (n > 0) {
        size_t diff = std::min(p->m_end - v, n);
        n -= diff;
        v += diff;
        m_size += diff;
        if (v == p->m_end && p->m_next != m_head) {
          p = p->m_next;
          v = p->m_begin;
        }
      }
      return *this;
    }

    iterator operator+(difference_type n) {
      if (n < 0) {
        return operator-(-n);
      } else {
        iterator tmp(*this);
        tmp.operator+=(n);
        return tmp;
      }
    }

    iterator& operator-=(difference_type n) { 
      while (n > 0) {
        size_t diff = std::min(v - p->m_begin, n);
        if (diff == 0) {
          assert(v == p->m_begin);
          operator--();
          --n;
          continue;
        }
        n -= diff;
        v -= diff;
        m_size -= diff;
      }
      return *this;
    }

    iterator operator-(difference_type n) { 
      if (n < 0) {
        return operator+(-n);
      } else {
        iterator tmp(*this);
        tmp.operator-=(n);
        return tmp;
      }
    }

    difference_type operator-(const iterator& o) {
      return static_cast<difference_type>(m_size) - static_cast<difference_type>(o.m_size);
    }

    reference operator[](int i) {
      iterator tmp(*this);
      tmp.operator+=(i);
      return *tmp;
    }

    const_reference operator[](int i) const {
      iterator tmp(*this);
      tmp.operator+=(i);
      return *tmp;
    }
  };

  typedef iterator const_iterator;

  const_iterator begin() const {
    return const_iterator(m_head, m_head, m_head ? m_head->m_begin : 0, 0);
  }

  iterator begin() {
    return iterator(m_head, m_head, m_head ? m_head->m_begin : 0, 0);
  }

  const_iterator end() const {
    if (m_head) {
      return const_iterator(m_head, m_head->m_prev, m_head->m_prev->m_end, m_size);
    } else {
      return const_iterator(m_head, 0, 0, m_size);
    }
  }

  iterator end() {
    if (m_head) {
      return iterator(m_head, m_head->m_prev, m_head->m_prev->m_end, m_size);
    } else {
      return iterator(m_head, 0, 0, m_size);
    }
  }
  
  reference push_back(const T& val) {
    Header* last;
    if (!m_head) {
      last = newHeader();
      insertBack(last);
    } else {
      last = m_head->m_prev;
    }
    if (last->m_end == last->m_last) {
      last = newHeader();
      insertBack(last);
    }
    T* rv = new (last->m_end) T(val);
    ++last->m_end;
    ++m_size;
    return *rv;
  }

  bool empty() const {
    return m_size == 0;
  }

  size_t size() const {
    return m_size;
  }

  void clear() {
    if (!m_head) {
      assert(m_size == 0);
      return;
    }

    // Clean up all but last node
    Header* h = m_head;
    while (h->m_next != m_head) {
      destroyRange(h->m_begin, h->m_end);
      Header* next = h->m_next;
      freeHeader(h);
      h = next;
    }

    // Clean up last node
    destroyRange(h->m_begin, h->m_end);
    freeHeader(h);
    m_head = 0;
    m_size = 0;
  }

  void pop_back() {
    Header *last = m_head->m_prev;
    assert(last->m_end != last->m_begin);
    assert(m_size > 0);
    --m_size;
    --last->m_end;
    last->m_end->~T();
    if (last->m_end == last->m_begin) {
      if (last == m_head) {
        // last node
        freeHeader(m_head);
        m_head = 0;
        assert(m_size == 0);
      } else {
        assert(last->m_next == m_head);
        last->m_prev->m_next = last->m_next;
        last->m_next->m_prev = last->m_prev;
        freeHeader(last);
      }
    }
  }

  void swap(Bag<T>& o) {
    if (isInlined())
      explode();
    if (o.isInlined())
      o.explode();

    std::swap(m_head, o.m_head);
    std::swap(m_size, o.m_size);
  }

  //! Add elements of other bag to this one, leaving other bag empty
  void splice(Bag<T>& o) {
    if (o.empty()) {
      return;
    } else if (empty()) {
      swap(o);
      return;
    }

    if (isInlined())
      explode();

    if (o.isInlined())
      o.explode();

    Header *last = m_head->m_prev;
    Header *olast = o.m_head->m_prev;

    last->m_next = o.m_head;
    o.m_head->m_prev = last;

    olast->m_next = m_head;
    m_head->m_prev = olast;

    m_size += o.m_size;
    o.m_head = 0;
    o.m_size = 0;
  }

  reference back() {
    return *(m_head->m_prev->m_end - 1);
  }

  const_reference back() const {
    return *(m_head->m_prev->m_end - 1);
  }
};

/**
 * Bag that supports a small number of inline elements before defaulting
 * to Bag<T> implementation.
 */
template<typename T, unsigned N>
class SmallBag: public Bag<T> {
  typedef typename Bag<T>::Header Header;
  typedef typename Bag<T>::U U;

  enum {
    //! Number of Us to cover N Ts
    MinUs = (static_cast<unsigned>(sizeof(T)) * N +
        static_cast<unsigned>(sizeof(U)) - 1) /
        static_cast<unsigned>(sizeof(U)),
    //! Number of actual elements given that the first element is stored in
    //! parent class, rounding up to avoid zero element array.
    ActualUs = MinUs > 1 ? (MinUs - 1) : 1,
    //! Number of Ts we have space for, which may be greater than N due to rounding
    MaxTs = (ActualUs + 1)* static_cast<unsigned>(sizeof(U)) / static_cast<unsigned>(sizeof(T))
  };

  U m_elements[ActualUs];

  Header m_h;

public:
  SmallBag() {
    T* a = static_cast<T*>(static_cast<void*>(&(this->m_first)));
    this->m_head = Bag<T>::initHeader(&m_h, a, a + MaxTs);
  }
};

/**
 * Like InsertBag but with random access iterators.
 */
// TODO(ddn): Remove need for explicit merge by adopting same techniques are InsBag
template<typename T>
class MergeBag: private boost::noncopyable {
  GaloisRuntime::PerThreadStorage<Bag<T> > bags;

public:
  typedef typename Bag<T>::value_type value_type;
  typedef typename Bag<T>::const_reference const_reference;
  typedef typename Bag<T>::reference reference;
  typedef typename Bag<T>::iterator iterator;
  typedef typename Bag<T>::const_iterator const_iterator;

  void merge() {
    Bag<T>& o = *bags.getRemote(0);
    for (unsigned i = 1; i < bags.size(); ++i) {
      o.splice(*bags.getRemote(i));
    }
  }

  const_iterator begin() const {
    return bags.getRemote(0)->begin();
  }

  iterator begin() {
    return bags.getRemote(0)->begin();
  }

  const_iterator end() const {
    return bags.getRemote(0)->end();
  }

  iterator end() {
    return bags.getRemote(0)->end();
  }
  
  reference push_back(const T& val) {
    return bags.getLocal()->push_back(val);
  }

  bool empty() const {
    return bags.getRemote(0)->empty();
  }

  size_t size() const {
    return bags.getRemote(0)->size();
  }

  void clear() {
    bags.getRemote(0)->clear();
  }

  void swap(MergeBag<T>& o) {
    for (unsigned i = 0; i < bags.size(); ++i) {
      std::swap(*bags.getRemote(i), *o.bags.getRemote(i));
    }
  }
};

template<typename T>
inline void swap(Galois::Bag<T>& a, Galois::Bag<T>& b) {
  a.swap(b);
}

template<typename T, unsigned N>
inline void swap(Galois::SmallBag<T,N>& a, Galois::SmallBag<T,N>& b) {
  a.swap(b);
}

template<typename T>
inline void swap(Galois::MergeBag<T>& a, Galois::MergeBag<T>& b) {
  a.swap(b);
}

}

#endif
