/** STL containers with Galois Allocators-*- C++ -*-
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
 * @section Description
 *
 * STL containers with Galois Allocators
 *
 * @author <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_GSTL_CONTAINERS_H
#define GALOIS_GSTL_CONTAINERS_H

namespace Galois {

namespace gstl {

  template<typename T>
  using Pow2Alloc = typename Runtime::MM::Pow_2_BlockAllocator<T>; 

  template<typename T>
  using FixedSizeAlloc = typename Runtime::MM::FixedSizeAllocator<T>; 

  template<typename T>
  using Vector = typename std::vector<T, typename Pow2Alloc<T> >; 

  template<typename T>
  using Deque = typename std::deque<T, typename Pow2Alloc<T> >; 

  template<typename T>
  using List = typename std::list<T, typename FixedSizeAlloc<T> >; 

  template<typename T, typename C>
  using Set = typename std::set<T, C, typename FixedSizeAlloc<T> >; 

  template<typename T, typename C>
  using PQ = MinHeap<T, C, typename Vector<T> >; 

  // template<typename T>
  // struct Pow2Alloc { typedef typename Runtime::MM::Pow_2_BlockAllocator<T> type; };
// 
  // template<typename T>
  // struct FixedSizeAlloc { typedef typename Runtime::MM::FixedSizeAllocator<T> type; };
// 
  // template<typename T>
  // struct Vector { typedef typename std::vector<T, typename Pow2Alloc<T>::type > type; };
// 
  // template<typename T>
  // struct Deque { typedef typename std::deque<T, typename Pow2Alloc<T>::type > type; };
// 
  // template<typename T>
  // struct List { typedef typename std::list<T, typename FixedSizeAlloc<T>::type > type; };
// 
  // template<typename T, typename C>
  // struct Set { typedef typename std::set<T, C, typename FixedSizeAlloc<T>::type > type; };
// 
  // template<typename T, typename C>
  // struct PQ { typedef MinHeap<T, C, typename Vector<T>::type > type; };
} // end namespace gstl


} // end namespace Galois


#endif //  GALOIS_GSTL_CONTAINERS_H
