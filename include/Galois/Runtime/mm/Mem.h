/** heap building blocks -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Strongly inspired by heap layers:
 *  http://www.heaplayers.org/
 * FSB is modified from:
 *  http://warp.povusers.org/FSBAllocator/
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_MEM_H
#define GALOIS_RUNTIME_MEM_H

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Substrate/SimpleLock.h"
#include "Galois/Substrate/PtrLock.h"
#include "Galois/Substrate/CacheLineStorage.h"

#include <boost/utility.hpp>
#include <cstdlib>
#include <cstring>
#include <map>
#include <list>
#include <cstddef>

#include <memory.h>

namespace Galois {
namespace Runtime {
//! Memory management functionality.
namespace MM {

extern size_t pageSize;
const size_t hugePageSize = 2*1024*1024;

void* pageAlloc();
void pageFree(void*);
//! Preallocate numpages large pages for each thread
void pagePreAlloc(int numpages);
//! Forces the given block to be paged into physical memory
void pageIn(void *buf, size_t len, size_t stride);
//! Forces the given readonly block to be paged into physical memory
void pageInReadOnly(void *buf, size_t len, size_t stride);

//! Returns total large pages allocated by Galois memory management subsystem
int numPageAllocTotal();
//! Returns total large pages allocated for thread by Galois memory management subsystem
int numPageAllocForThread(unsigned tid);

//! Returns total small pages allocated by OS on a NUMA node
int numNumaAllocForNode(unsigned nodeid);
//! Returns number of NUMA nodes on machine
int numNumaNodes();

/**
 * Allocates memory interleaved across NUMA nodes. 
 * 
 * If full, allocate across all NUMA nodes; otherwise,
 * allocate across NUMA nodes corresponding to active
 * threads.
 */
void* largeInterleavedAlloc(size_t bytes, bool full = true);
//! Frees memory allocated by {@link largeInterleavedAlloc()}
void largeInterleavedFree(void* mem, size_t bytes);

//! Allocates a large block of memory
void* largeAlloc(size_t bytes, bool preFault = true);
//! Frees memory allocated by {@link largeAlloc()}
void largeFree(void* mem, size_t bytes);

//! Print lines from /proc/pid/numa_maps that contain at least n (non-huge) pages
void printInterleavedStats(int minPages = 16*1024);

//! [Example Third Party Allocator]
class MallocHeap {
public:
  //! Supported allocation size in bytes. If 0, heap supports variable sized allocations
  enum { AllocSize = 0 };

  void* allocate(size_t size) {
    return malloc(size);
  }
  
  void deallocate(void* ptr) {
    free(ptr);
  }
};
//! [Example Third Party Allocator]


//! Per-thread heaps using Galois thread aware construct
template<class SourceHeap>
class ThreadPrivateHeap {
  PerThreadStorage<SourceHeap> heaps;

public:
  enum { AllocSize = SourceHeap::AllocSize };

  ThreadPrivateHeap() {}
  ~ThreadPrivateHeap() {
    clear();
  }

  template<typename... Args>
  inline void* allocate(size_t size, Args&&... args) {
    return heaps.getLocal()->allocate(size, std::forward<Args>(args)...);
  }

  inline void deallocate(void* ptr) {
    heaps.getLocal()->deallocate(ptr);
  }

  void clear() {
    for (unsigned int i = 0; i < heaps.size(); i++)
      heaps.getRemote(i)->clear();
  }
};

//! Apply a lock to a heap
template<class SourceHeap>
class LockedHeap : public SourceHeap {
  Substrate::SimpleLock lock;

public:
  enum { AllocSize = SourceHeap::AllocSize };

  inline void* allocate(size_t size) {
    lock.lock();
    void* retval = SourceHeap::allocate(size);
    lock.unlock();
    return retval;
  }
  
  inline void deallocate(void* ptr) {
    lock.lock();
    SourceHeap::deallocate(ptr);
    lock.unlock();
  }
};

template<typename SourceHeap>
class ZeroOut : public SourceHeap {
public:
  enum { AllocSize = SourceHeap::AllocSize };

  inline void* allocate(size_t size) {
    void* retval = SourceHeap::allocate(size);
    memset(retval, 0, size);
    return retval;
  }

  inline void deallocate(void* ptr) {
    SourceHeap::deallocate(ptr);
  }
};

//! Add a header to objects
template<typename Header, typename SourceHeap>
class AddHeader : public SourceHeap {
  enum { offset = (sizeof(Header) + (sizeof(double) - 1)) & ~(sizeof(double) - 1) };

public:
  inline void* allocate(size_t size) {
    //First increase the size of the header to be aligned to a double
    void* ptr = SourceHeap::allocate(size + offset);
    //Now return the offseted pointer
    return (char*)ptr + offset;
  }
  
  inline void deallocate(void* ptr) {
    SourceHeap::deallocate(getHeader(ptr));
  }

  inline static Header* getHeader(void* ptr) {
    return (Header*)((char*)ptr - offset);
  }
};

//! Allow looking up parent heap pointers
template<class SourceHeap>
class OwnerTaggedHeap : public AddHeader<void*, SourceHeap> {
  typedef AddHeader<OwnerTaggedHeap*, SourceHeap> Src;

public:
  inline void* allocate(size_t size) {
    void* retval = Src::allocate(size);
    *(Src::getHeader(retval)) = this;
    return retval;
  }

  inline void deallocate(void* ptr) {
    assert(*(Src::getHeader(ptr)) == this);
    Src::deallocate(ptr);
  }

  inline static OwnerTaggedHeap* owner(void* ptr) {
    return *(OwnerTaggedHeap**)Src::getHeader(ptr);
  }
};

//! Maintain a freelist
template<class SourceHeap>
class FreeListHeap : public SourceHeap {
  struct FreeNode {
    FreeNode* next;
  };
  FreeNode* head;

public:
  enum { AllocSize = SourceHeap::AllocSize };

  void clear() {
    while (head) {
      FreeNode* N = head;
      head = N->next;
      SourceHeap::deallocate(N);
    }
  }

  FreeListHeap() : head(0) {}
  ~FreeListHeap() {
    clear();
  }

  inline void* allocate(size_t size) {
    if (head) {
      void* ptr = head;
      head = head->next;
      return ptr;
    }
    return SourceHeap::allocate(size);
  }

  inline void deallocate(void* ptr) {
    if (!ptr) return;
    assert((uintptr_t)ptr > 0x100);
    FreeNode* NH = (FreeNode*)ptr;
    NH->next = head;
    head = NH;
  }
};

//! Maintain a freelist using a lock which doesn't cover SourceHeap
template<class SourceHeap>
class SelfLockFreeListHeap : public SourceHeap {
  struct FreeNode {
    FreeNode* next;
  };
  FreeNode* head;

public:
  enum { AllocSize = SourceHeap::AllocSize };

  void clear() {
    FreeNode* h = 0;
    do {
      h = head;
    } while (!__sync_bool_compare_and_swap(&head, h, 0));
    while (h) {
      FreeNode* N = h;
      h = N->next;
      SourceHeap::deallocate(N);
    }
  }

  SelfLockFreeListHeap() : head(0) {}
  ~SelfLockFreeListHeap() {
    clear();
  }

  inline void* allocate(size_t size) {
    static Substrate::SimpleLock lock;

    lock.lock();
    FreeNode* OH = 0;
    FreeNode* NH = 0;
    do {
      OH = head;
      if (!OH) {
	lock.unlock();
	return SourceHeap::allocate(size);
      }
      NH = OH->next; //The lock protects this line
    } while (!__sync_bool_compare_and_swap(&head, OH, NH));
    lock.unlock();
    assert(OH);
    return (void*)OH;
  }

  inline void deallocate(void* ptr) {
    if (!ptr) return;
    FreeNode* OH;
    FreeNode* NH;
    do {
      OH = head;
      NH = (FreeNode*)ptr;
      NH->next = OH;
    } while (!__sync_bool_compare_and_swap(&head, OH, NH));
  }
};


template<unsigned ElemSize, typename SourceHeap>
class BlockHeap : public SourceHeap {
  struct TyEq {
    double data[((ElemSize + sizeof(double) - 1) & ~(sizeof(double) - 1))/sizeof(double)];
  };

  struct Block_basic {
    union {
      Block_basic* next;
      double dummy;
    };
    TyEq data[1];
  };

  enum {
    BytesLeft = (SourceHeap::AllocSize - sizeof(Block_basic)),
    BytesLeftR = BytesLeft & ~(sizeof(double) - 1),
    FitLeft = BytesLeftR / sizeof(TyEq[1]),
    TotalFit = FitLeft + 1
  };

  struct Block {
    union {
      Block* next;
      double dummy;
    };
    TyEq data[TotalFit];
  };

  Block* head;
  int headIndex;

  void refill() {
    void* P = SourceHeap::allocate(SourceHeap::AllocSize);
    Block* BP = (Block*)P;
    BP->next = head;
    head = BP;
    headIndex = 0;
  }

public:
  enum { AllocSize = ElemSize };

  void clear() {
    while (head) {
      Block* B = head;
      head = B->next;
      SourceHeap::deallocate(B);
    }
  }

  BlockHeap() :SourceHeap(), head(0), headIndex(0) {
    static_assert(sizeof(Block) <= SourceHeap::AllocSize, "");
  }

  ~BlockHeap() {
    clear();
  }

  inline void* allocate(size_t size) {
    assert(size == ElemSize);
    if (!head || headIndex == TotalFit)
      refill();
    return &head->data[headIndex++];
  }

  inline void deallocate(void* ptr) {}
};

//! This implements a bump pointer though chunks of memory
template<typename SourceHeap>
class BumpHeap : public SourceHeap {
  struct Block {
    union {
      Block* next;
      double dummy; // for alignment
    };
  };

  Block* head;
  int offset;

  void refill() {
    void* P = SourceHeap::allocate(SourceHeap::AllocSize);
    Block* BP = (Block*)P;
    BP->next = head;
    head = BP;
    offset = sizeof(Block);
  }

public:
  enum { AllocSize = 0 };

  BumpHeap(): SourceHeap(), head(0), offset(0) {}

  ~BumpHeap() {
    clear();
  }

  void clear() {
    while (head) {
      Block* B = head;
      head = B->next;
      SourceHeap::deallocate(B);
    }
  }

  inline void* allocate(size_t size) {
    // Increase to alignment
    size_t alignedSize = (size + sizeof(double) - 1) & ~(sizeof(double) - 1);
    // Check current block
    if (!head || offset + alignedSize > SourceHeap::AllocSize) {
      refill();
    }
    if (offset + alignedSize > SourceHeap::AllocSize) {
      std::abort(); // TODO: remove
      throw std::bad_alloc();
    }
    char* retval = (char*)head;
    retval += offset;
    offset += alignedSize;
    return retval;
  }

  /**
   * Allocates size bytes but may fail. If so, size < allocated and
   * allocated is the number of bytes allocated in the returned buffer.
   */
  inline void* allocate(size_t size, size_t& allocated) {
    // Increase to alignment
    size_t alignedSize = (size + sizeof(double) - 1) & ~(sizeof(double) - 1);
    if (alignedSize > SourceHeap::AllocSize) {
      alignedSize = SourceHeap::AllocSize;
    }
    // Check current block
    if (!head || offset + alignedSize > SourceHeap::AllocSize) {
      size_t remaining = SourceHeap::AllocSize - offset;
      assert((remaining & (sizeof(double) - 1)) == 0); // should still be aligned
      if (!remaining) {
        refill();
      } else {
        alignedSize = remaining;
      }
    }
    char* retval = (char*)head;
    retval += offset;
    offset += alignedSize;
    allocated = (alignedSize > size) ? size : alignedSize;
    return retval;
  }

  inline void deallocate(void* ptr) {}
};

/**
 * This implements a bump pointer though chunks of memory that falls back
 * to malloc if the source heap cannot accommodate an allocation.
 */
template<typename SourceHeap>
class BumpWithMallocHeap : public SourceHeap {
  struct Block {
    union {
      Block* next;
      double dummy; // for alignment
    };
  };

  Block* head;
  Block* fallbackHead;
  int offset;

  //! Given block of memory P, update head pointer and offset metadata
  void refill(void* P, Block*& h, int* o) {
    Block* BP = (Block*)P;
    BP->next = h;
    h = BP;
    if (o)
      *o = sizeof(Block);
  }

public:
  enum { AllocSize = 0 };

  BumpWithMallocHeap(): SourceHeap(), head(0), fallbackHead(0), offset(0) { }

  ~BumpWithMallocHeap() {
    clear();
  }

  void clear() {
    while (head) {
      Block* B = head;
      head = B->next;
      SourceHeap::deallocate(B);
    }
    while (fallbackHead) {
      Block* B = fallbackHead;
      fallbackHead = B->next;
      free(B);
    }
  }

  inline void* allocate(size_t size) {
    // Increase to alignment
    size_t alignedSize = (size + sizeof(double) - 1) & ~(sizeof(double) - 1);
    if (sizeof(Block) + alignedSize > SourceHeap::AllocSize) {
      void* p = malloc(alignedSize + sizeof(Block));
      refill(p, fallbackHead, NULL);
      return (char*)p + sizeof(Block);
    }
    // Check current block
    if (!head || offset + alignedSize > SourceHeap::AllocSize)
      refill(SourceHeap::allocate(SourceHeap::AllocSize), head, &offset);
    char* retval = (char*)head;
    retval += offset;
    offset += alignedSize;
    return retval;
  }

  inline void deallocate(void* ptr) {}
};

//! This is the base source of memory for all allocators.
//! It maintains a freelist of hunks acquired from the system
class SystemHeap {
public:
  enum { AllocSize = hugePageSize };

  SystemHeap();
  ~SystemHeap();

  inline void* allocate(size_t size) {
    return pageAlloc();
  }

  inline void deallocate(void* ptr) {
    pageFree(ptr);
  }
};

#ifdef GALOIS_FORCE_STANDALONE
class SizedHeapFactory: private boost::noncopyable {
public:
  typedef MallocHeap SizedHeap;

  static SizedHeap* getHeapForSize(const size_t) {
    return &alloc;
  }

private:
  static SizedHeap alloc;
};
#else
class SizedHeapFactory: private boost::noncopyable {
public:
//! [FixedSizeAllocator example]
  typedef ThreadPrivateHeap<
    FreeListHeap<BumpHeap<SystemHeap> > > SizedHeap;
//! [FixedSizeAllocator example]

  static SizedHeap* getHeapForSize(const size_t);

private:
  typedef std::map<size_t, SizedHeap*> HeapMap;
  static SizedHeapFactory* getInstance();
  static Substrate::PtrLock<SizedHeapFactory> instance;
  static __thread HeapMap* localHeaps;
  HeapMap heaps;
  std::list<HeapMap*> allLocalHeaps;
  Substrate::SimpleLock lock;

  SizedHeapFactory();
  ~SizedHeapFactory();

  SizedHeap* getHeap(const size_t);
};
#endif

/**
 * Scalable variable-size allocations.
 *
 * Slight misnomer as this doesn't support allocations greater than a page.
 * Users should call {@link allocate(size_t, size_t&)} multiple times to split
 * large allocations over multiple pages.
 */
struct VariableSizeHeap: public ThreadPrivateHeap<BumpHeap<SystemHeap>> {
  enum { AllocSize = 0 };
};

//! Main scalable allocator in Galois
class FixedSizeHeap {
  SizedHeapFactory::SizedHeap* heap;

public:
  FixedSizeHeap(size_t size) {
    heap = SizedHeapFactory::getHeapForSize(size);
  }

  inline void* allocate(size_t size) {
    return heap->allocate(size);
  }

  inline void deallocate(void* ptr) {
    heap->deallocate(ptr);
  }

  inline bool operator!=(const FixedSizeHeap& rhs) const {
    return heap != rhs.heap;
  }
  
  inline bool operator==(const FixedSizeHeap& rhs) const {
    return heap == rhs.heap;
  }
};

class SerialNumaHeap {
  enum { offset = (sizeof(size_t) + (sizeof(double) - 1)) & ~(sizeof(double) - 1) };

public:
  enum { AllocSize = 0 };

  void* allocate(size_t size) {
    char* ptr = (char*) largeInterleavedAlloc(size + offset, true);
    size_t* header = (size_t*) ptr;
    *header = size;
    return ptr + offset;
  }

  void deallocate(void* ptr) {
    char* realPtr = ((char*)ptr - offset);
    size_t* header = (size_t*) realPtr;
    largeInterleavedFree(header, *header);
  }
};


////////////////////////////////////////////////////////////////////////////////
// Now adapt to standard std allocators
////////////////////////////////////////////////////////////////////////////////

//!A fixed size block allocator
template<typename Ty>
class FixedSizeAllocator;

template<>
class FixedSizeAllocator<void> {
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  template<typename Other>
  struct rebind { typedef FixedSizeAllocator<Other> other; };
};

template<typename Ty>
class FixedSizeAllocator {
  inline void destruct(char*) const { }
  inline void destruct(wchar_t*) const { }
  template<typename T> inline void destruct(T* t) const { t->~T(); }

  FixedSizeHeap heap;

public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef Ty *pointer;
  typedef const Ty *const_pointer;
  typedef Ty& reference;
  typedef const Ty& const_reference;
  typedef Ty value_type;
  
  template<class Other>
  struct rebind { typedef FixedSizeAllocator<Other> other; };

  FixedSizeAllocator() throw(): heap(sizeof(Ty)) {}
  template <class U> FixedSizeAllocator(const FixedSizeAllocator<U>&) throw(): heap(sizeof(Ty)) {}

  inline pointer address(reference val) const { return &val; }
  inline const_pointer address(const_reference val) const { return &val; }

  pointer allocate(size_type size) {
    if (size > max_size())
      throw std::bad_alloc();
    return static_cast<pointer>(heap.allocate(sizeof(Ty)));
  }
  
  void deallocate(pointer ptr, size_type len) {
    assert(len == 1);
    heap.deallocate(ptr);
  }
  
  template<class U, class... Args>
  inline void construct(U* p, Args&&... args ) const {
    ::new((void*)p) U(std::forward<Args>(args)...);
  }
  
  inline void destroy(pointer ptr) const {
    destruct(ptr);
  }

  size_type max_size() const throw() { return 1; }

  template<typename T1>
  inline bool operator!=(const FixedSizeAllocator<T1>& rhs) const {
    return heap != rhs.heap;
  }

  template<typename T1>
  inline bool operator==(const FixedSizeAllocator<T1>& rhs) const {
    return heap == rhs.heap;
  }
};

class Pow_2_BlockHeap: private boost::noncopyable {

  private:

  static const bool USE_MALLOC_AS_BACKUP = true;

  static const size_t LOG2_MIN_SIZE = 3; // 2^3 == 8 bytes
  static const size_t LOG2_MAX_SIZE = 16; // 64k

  typedef FixedSizeHeap Heap_ty;

  std::vector<Heap_ty> heapTable;

  static Substrate::PtrLock<Pow_2_BlockHeap> instance;

  static inline size_t pow2 (unsigned i) {
    return (1U << i);
  }

  static unsigned nextLog2 (const size_t allocSize) {

    unsigned i = LOG2_MIN_SIZE;

    while (pow2 (i) < allocSize) {
      ++i;
    }

    // if (pow2 (i) > pow2 (LOG2_MAX_SIZE)) {
      // std::fprintf (stderr, "ERROR: block bigger than huge page size requested\n");
      // throw std::bad_alloc();
    // }

    return i;
  }


  void populateTable (void) {
    assert (heapTable.empty ());

    heapTable.clear ();
    for (unsigned i = 0; i <= LOG2_MAX_SIZE; ++i) {
      heapTable.push_back (Heap_ty (pow2 (i)));
    }
  }

  Pow_2_BlockHeap (void) throw ();


  public:

  static Pow_2_BlockHeap* getInstance (void);

  void* allocateBlock (const size_t allocSize) {

    if (allocSize > pow2 (LOG2_MAX_SIZE)) {
      if (USE_MALLOC_AS_BACKUP) {
        return malloc (allocSize);
      } else {
        std::fprintf (stderr, "ERROR: block bigger than huge page size requested\n");
        throw std::bad_alloc();
      }
    } else {

      unsigned i = nextLog2 (allocSize);
      assert (i < heapTable.size());
      return heapTable[i].allocate (pow2 (i));
    }
  }

  void deallocateBlock (void* ptr, const size_t allocSize) {
    if (allocSize > pow2 (LOG2_MAX_SIZE)) {
      if (USE_MALLOC_AS_BACKUP) {
        free (ptr);
      } else {
        std::fprintf (stderr, "ERROR: block bigger than huge page size requested\n");
        throw std::bad_alloc();
      }
    } else {
      unsigned i = nextLog2 (allocSize);
      assert (i < heapTable.size());
      heapTable[i].deallocate (ptr);
    }
  }
};

template <typename Ty>
class Pow_2_BlockAllocator {


  template<typename T> 
  static inline void destruct(T* t) { 
    if (!std::is_scalar<T>::value) {
      t->~T(); 
    }
  }


public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef Ty *pointer;
  typedef const Ty *const_pointer;
  typedef Ty& reference;
  typedef const Ty& const_reference;
  typedef Ty value_type;
  
  template<class Other>
  struct rebind { typedef Pow_2_BlockAllocator<Other> other; };

  Pow_2_BlockHeap* heap;

  Pow_2_BlockAllocator() throw(): heap (Pow_2_BlockHeap::getInstance ()) {
  }

  // template <typename U>
  // friend class Pow_2_BlockAllocator<U>;

  template <typename U> 
  Pow_2_BlockAllocator(const Pow_2_BlockAllocator<U>& that) throw()
  : heap (that.heap) {}

  inline pointer address(reference val) const { return &val; }

  inline const_pointer address(const_reference val) const { return &val; }

  pointer allocate(size_type size) {
    return static_cast<pointer>(heap->allocateBlock (size * sizeof (Ty)));
  }
  
  void deallocate(pointer ptr, size_type len) {
    heap->deallocateBlock(ptr, len * sizeof(Ty));
  }
  
  template<class U, class... Args>
  inline void construct(U* p, Args&&... args ) const {
    ::new((void*)p) U(std::forward<Args>(args)...);
  }
  
  inline void destroy(pointer ptr) const {
    destruct (ptr);
  }

  size_type max_size() const throw() { return size_type (-1); }
};

template <>
class Pow_2_BlockAllocator<void> {
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  template<typename Other>
  struct rebind { typedef Pow_2_BlockAllocator<Other> other; };
};

//! Keep a reference to an external allocator
template<typename Ty, typename HeapTy>
class ExternalHeapAllocator;

template<typename HeapTy>
class ExternalHeapAllocator<void, HeapTy> {
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  template<typename Other>
  struct rebind { typedef ExternalHeapAllocator<Other,HeapTy> other; };
};

template<typename Ty, typename HeapTy>
class ExternalHeapAllocator {
  inline void destruct(char*) const {}
  inline void destruct(wchar_t*) const { }
  template<typename T> inline void destruct(T* t) const { t->~T(); }

public:
  HeapTy* heap; // Should be private except that makes copy hard

  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef Ty *pointer;
  typedef const Ty *const_pointer;
  typedef Ty& reference;
  typedef const Ty& const_reference;
  typedef Ty value_type;
  
  template<class Other>
  struct rebind {
    typedef ExternalHeapAllocator<Other, HeapTy> other;
  };

  explicit ExternalHeapAllocator(HeapTy* a) throw(): heap(a) {}

  template<class T1>
  ExternalHeapAllocator(const ExternalHeapAllocator<T1,HeapTy>& rhs) throw() {
    heap = rhs.heap;
  }
  
  inline pointer address(reference val) const { return &val; }

  inline const_pointer address(const_reference val) const { return &val; }
  
  pointer allocate(size_type size) {
    if (size > max_size())
      throw std::bad_alloc();
    return static_cast<pointer>(heap->allocate(size*sizeof(Ty)));
  }
  
  void deallocate(pointer ptr, size_type len) {
    heap->deallocate(ptr);
  }
  
  inline void construct(pointer ptr, const_reference val) const {
    new (ptr) Ty(val);
  }

  template<class U, class... Args >
  inline void construct(U* p, Args&&... args ) const {
    ::new((void*)p) U(std::forward<Args>(args)...);
  }
  
  void destroy(pointer ptr) const {
    destruct(ptr);
  }
  
  size_type max_size() const throw() { return (HeapTy::AllocSize == 0) ? size_t(-1)/sizeof(Ty) : HeapTy::AllocSize/sizeof(Ty); }

  template<typename T1,typename A1>
  bool operator!=(const ExternalHeapAllocator<T1,A1>& rhs) const {
    return heap != rhs.heap;
  }

  template<typename T1,typename A1>
  bool operator==(const ExternalHeapAllocator<T1,A1>& rhs) const {
    return heap == rhs.heap;
  }
};

template<typename T>
class SerialNumaAllocator: public ExternalHeapAllocator<T, SerialNumaHeap> {
  using Super = ExternalHeapAllocator<T, SerialNumaHeap>;
  SerialNumaHeap heap;

public:
  template<class Other>
  struct rebind {
    typedef SerialNumaAllocator<Other> other;
  };

  SerialNumaAllocator(): Super(&heap) {}
};

} // end namespace MM
} // end namespace Runtime
} // end namespace Galois

#endif
