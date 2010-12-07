// -*- C++ -*-

/*===========================================================================
  This library is released under the MIT license. See FSBAllocator.html
  for further information and documentation.

Copyright (c) 2008 Juha Nieminen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
=============================================================================*/

//This is a modification and specialization of FSBAllocator to Galois



#ifndef INCLUDE_FSBALLOCATOR_HH
#define INCLUDE_FSBALLOCATOR_HH

#include <new>
#include <cassert>
#include <vector>

#include "Support/ThreadSafe/simple_lock.h"
#include "mmap.h"

namespace GaloisRuntime {
namespace MM {

template<unsigned ElemSize, bool ts>
class FSBAllocator_ElemAllocator : private threadsafe::simpleLock<int, ts>
{
  typedef std::size_t Data_t;
  static const Data_t BlockElements = 512;

  static const Data_t DSize = sizeof(Data_t);
  static const Data_t ElemSizeInDSize = (ElemSize + (DSize-1)) / DSize;
  static const Data_t UnitSizeInDSize = ElemSizeInDSize + 1;
  static const Data_t BlockSize = BlockElements*UnitSizeInDSize;

  using threadsafe::simpleLock<int, ts>::lock;
  using threadsafe::simpleLock<int, ts>::unlock;

  class MemBlock
  {
    Data_t* block;
    Data_t firstFreeUnitIndex, allocatedElementsAmount, endIndex;

  public:
    MemBlock():
      block(0),
      firstFreeUnitIndex(Data_t(-1)),
      allocatedElementsAmount(0)
    {}

    bool isFull() const
    {
      return allocatedElementsAmount == BlockElements;
    }

    void clear()
    {
      delete[] block;
      block = 0;
      firstFreeUnitIndex = Data_t(-1);
    }

    void* allocate(Data_t vectorIndex)
    {
      if(firstFreeUnitIndex == Data_t(-1))
	{
	  if(!block)
	    {
	      block = new Data_t[BlockSize];
	      if(!block) return 0;
	      endIndex = 0;
	    }

	  Data_t* retval = block + endIndex;
	  endIndex += UnitSizeInDSize;
	  retval[ElemSizeInDSize] = vectorIndex;
	  ++allocatedElementsAmount;
	  return retval;
	}
      else
	{
	  Data_t* retval = block + firstFreeUnitIndex;
	  firstFreeUnitIndex = *retval;
	  ++allocatedElementsAmount;
	  return retval;
	}
    }

    void deallocate(Data_t* ptr)
    {
      *ptr = firstFreeUnitIndex;
      firstFreeUnitIndex = ptr - block;

      if(--allocatedElementsAmount == 0)
	clear();
    }
  };

  struct BlocksVector
  {
    std::vector<MemBlock> data;

    BlocksVector() { data.reserve(1024); }

    ~BlocksVector()
    {
      for(size_t i = 0; i < data.size(); ++i)
	data[i].clear();
    }
  };

  static BlocksVector blocksVector;
  static std::vector<Data_t> blocksWithFree;

public:
  static void* allocate()
  {
    lock();

    if(blocksWithFree.empty())
      {
	blocksWithFree.push_back(blocksVector.data.size());
	blocksVector.data.push_back(MemBlock());
      }

    const Data_t index = blocksWithFree.back();
    MemBlock& block = blocksVector.data[index];
    void* retval = block.allocate(index);

    if(block.isFull())
      blocksWithFree.pop_back();

    unlock();

    return retval;
  }

  static void deallocate(void* ptr)
  {
    if(!ptr) return;

    lock();

    Data_t* unitPtr = (Data_t*)ptr;
    const Data_t blockIndex = unitPtr[ElemSizeInDSize];
    MemBlock& block = blocksVector.data[blockIndex];

    if(block.isFull())
      blocksWithFree.push_back(blockIndex);
    block.deallocate(unitPtr);
    
    unlock();
  }
};

template<unsigned ElemSize, bool ts>
typename FSBAllocator_ElemAllocator<ElemSize, ts>::BlocksVector
FSBAllocator_ElemAllocator<ElemSize, ts>::blocksVector;

template<unsigned ElemSize, bool ts>
std::vector<typename FSBAllocator_ElemAllocator<ElemSize, ts>::Data_t>
FSBAllocator_ElemAllocator<ElemSize, ts>::blocksWithFree;

template<unsigned ElemSize, class BaseAlloc = mmapWrapper>
class FSBAllocator2_ElemAllocator {
  static const size_t DSize = sizeof(size_t);
  static const size_t ElemSizeInDSize = (ElemSize + (DSize-1)) / DSize;
  static const size_t BlockSize = (BaseAlloc::AllocSize / (DSize * ElemSizeInDSize)) * ElemSizeInDSize;
  static const size_t BlockElements = BlockSize / ElemSizeInDSize;
  
  struct Blocks {
    std::vector<size_t*> ptrs;
    
    Blocks() {
      ptrs.reserve(256);
      ptrs.push_back((size_t*)BaseAlloc::alloc());
    }

    ~Blocks() {
      for(size_t i = 0; i < ptrs.size(); ++i)
	BaseAlloc::free(ptrs[i]);
    }
  };

  Blocks blocks;
  size_t headIndex;
  size_t* freeList;
  size_t allocatedElementsAmount;

  void freeAll() {
    for(size_t i = 1; i < blocks.ptrs.size(); ++i)
      BaseAlloc::free(blocks.ptr[i]);
    blocks.ptrs.resize(1);
    headIndex = 0;
    freeList = 0;
  }

public:
  FSBAllocator2_ElemAllocator() :headIndex(0), freeList(0), allocatedElementsAmount(0) {}

  static void merge(FSBAllocator2_ElemAllocator& lhs,
		    FSBAllocator2_ElemAllocator& rhs) {
    lhs.blocks.ptrs.insert(lhs.blocks.ptrs.begin(),
			   rhs.blocks.ptrs.begin(),
			   rhs.blocks.ptrs.end());
    lhs.blocks.ptrs.clear();
  }

  void* allocate() {
    ++allocatedElementsAmount;

    if(freeList) {
      size_t* retval = freeList;
      freeList = reinterpret_cast<size_t*>(*freeList);
      return retval;
    }

    if(headIndex == BlockSize) {
      blocks.ptrs.push_back((size_t*)BaseAlloc::alloc());
      headIndex = 0;
    }

    size_t* retval = &(blocks.ptrs.back()[headIndex]);
    headIndex += ElemSizeInDSize;
    
    return retval;
  }

  void deallocate(void* ptr) {
    if(ptr) {
      size_t* sPtr = (size_t*)ptr;
      *sPtr = reinterpret_cast<size_t>(freeList);
      freeList = sPtr;
      
      if(--allocatedElementsAmount == 0)
	freeAll();
    }
  }

  void cleanSweep(size_t unusedValue = size_t(-1)) {
    while(freeList) {
      size_t* current = freeList;
      freeList = reinterpret_cast<size_t*>(*freeList);
      *current = unusedValue;
    }
    
    for(size_t i = headIndex; i < BlockSize; i += ElemSizeInDSize)
      blocks.ptrs.back()[i] = unusedValue;
    
    for(size_t blockInd = 1; blockInd < blocks.ptrs.size();) {
      size_t* block = blocks.ptrs[blockInd];
      size_t freeAmount = 0;
      for(size_t i = 0; i < BlockSize; i += ElemSizeInDSize)
	if(block[i] == unusedValue)
	  ++freeAmount;
      
      if(freeAmount == BlockElements) {
	delete[] block;
	blocks.ptrs[blockInd] = blocks.ptrs.back();
	blocks.ptrs.pop_back();
      }
      else ++blockInd;
    }
    
    const size_t* lastBlock = blocks.ptrs.back();
    for(headIndex = BlockSize; headIndex > 0; headIndex -= ElemSizeInDSize)
      if(lastBlock[headIndex-ElemSizeInDSize] != unusedValue)
	break;
    
    const size_t lastBlockIndex = blocks.ptrs.size() - 1;
    for(size_t blockInd = 0; blockInd <= lastBlockIndex; ++blockInd) {
      size_t* block = blocks.ptrs[blockInd];
      for(size_t i = 0; i < BlockSize; i += ElemSizeInDSize) {
	if(blockInd == lastBlockIndex && i == headIndex)
	  break;
	
	if(block[i] == unusedValue)
	  deallocate(block + i);
      }
    }
  }
};

template<typename Ty, bool ts>
class FSBAllocator
{
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef Ty *pointer;
  typedef const Ty *const_pointer;
  typedef Ty& reference;
  typedef const Ty& const_reference;
  typedef Ty value_type;

  pointer address(reference val) const { return &val; }
  const_pointer address(const_reference val) const { return &val; }

  template<class Other>
  struct rebind
  {
    typedef FSBAllocator<Other, ts> other;
  };

  FSBAllocator() throw() {}

  template<class Other, bool ts2>
  FSBAllocator(const FSBAllocator<Other, ts2>&) throw() {}

  template<class Other, bool ts2>
  FSBAllocator& operator=(const FSBAllocator<Other, ts2>&) { return *this; }

  pointer allocate(size_type count, const void* = 0)
  {
    assert(count == 1);
    return static_cast<pointer>
      (FSBAllocator_ElemAllocator<sizeof(Ty), ts>::allocate());
  }

  void deallocate(pointer ptr, size_type)
  {
    FSBAllocator_ElemAllocator<sizeof(Ty), ts>::deallocate(ptr);
  }

  void construct(pointer ptr, const Ty& val)
  {
    new ((void *)ptr) Ty(val);
  }

  void destroy(pointer ptr)
  {
    ptr->Ty::~Ty();
  }

  size_type max_size() const throw() { return 1; }
};


template<typename Ty>
class FSBAllocator2
{
  GaloisRuntime::CPUSpaced<FSBAllocator2_ElemAllocator<sizeof(Ty)> > AllocList;

public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef Ty *pointer;
  typedef const Ty *const_pointer;
  typedef Ty& reference;
  typedef const Ty& const_reference;
  typedef Ty value_type;
  
  pointer address(reference val) const { return &val; }
  const_pointer address(const_reference val) const { return &val; }
  
  template<class Other>
  struct rebind
  {
    typedef FSBAllocator2<Other> other;
  };
  
  FSBAllocator2() throw() 
  :AllocList(FSBAllocator2_ElemAllocator<sizeof(Ty)>::merge)
  {}
  
  pointer allocate()
  {
    return static_cast<pointer>
      (AllocList.get().allocate());
  }
  
  void deallocate(pointer ptr, size_type)
  {
    AllocList.get().deallocate(ptr);
  }
  
  template<typename TyC>
  void construct(pointer ptr, const TyC& val)
  {
    new ((void *)ptr) Ty(val);
  }
  
  void destroy(pointer ptr)
  {
    ptr->Ty::~Ty();
  }
  
  size_type max_size() const throw() { return 1; }
  
  void cleanSweep(size_t unusedValue = size_t(-1))
  {
    AllocList.get().cleanSweep(unusedValue);
  }
};

}
}

#endif
