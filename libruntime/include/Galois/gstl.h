/** Simple STL style algorithms, STL data structures with Galois allocators -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Loc Hoang <l_hoang@utexas.edu> (prefix range)
 */
#ifndef GALOIS_GSTL_H
#define GALOIS_GSTL_H

#include "PriorityQueue.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <cassert>


#include <vector>
#include <set>
#include <deque>
#include <list>

namespace Galois {

namespace gstl {

  template<typename T>
  using Pow2Alloc = typename Runtime::Pow_2_BlockAllocator<T>; 

  template<typename T>
  using FixedSizeAlloc = typename Runtime::FixedSizeAllocator<T>; 

  template<typename T>
  using Vector = std::vector<T, Pow2Alloc<T> >; 

  template<typename T>
  using Deque = std::deque<T, Pow2Alloc<T> >; 

  template<typename T>
  using List = std::list<T, FixedSizeAlloc<T> >; 

  template<typename T, typename C=std::less<T> >
  using Set = std::set<T, C, FixedSizeAlloc<T> >; 

  template<typename T, typename C=std::less<T> >
  using PQ = MinHeap<T, C, Vector<T> >; 

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


template<typename IterTy, class Distance>
IterTy safe_advance_dispatch(IterTy b, IterTy e, Distance n, std::random_access_iterator_tag) {
  if (std::distance(b,e) >= n)
    return b + n;
  else
    return e;
}

template<typename IterTy, class Distance>
IterTy safe_advance_dispatch(IterTy b, IterTy e, Distance n, std::input_iterator_tag) {
  while (b != e && n--)
    ++b;
  return b;
}

/**
 * Like std::advance but returns end if end is closer than the advance amount.
 */
template<typename IterTy, class Distance>
IterTy safe_advance(IterTy b, IterTy e, Distance n) {
  typename std::iterator_traits<IterTy>::iterator_category category;
  return safe_advance_dispatch(b,e,n,category);
}


/**
 * Finds the midpoint of a range.  The first half is always be bigger than
 * the second half if the range has an odd length.
 */
template<typename IterTy>
IterTy split_range(IterTy b, IterTy e) {
  std::advance(b, (std::distance(b,e) + 1) / 2);
  return b;
}

/**
 * Returns a continuous block from the range based on the number of
 * divisions and the id of the block requested
 */
template<typename IterTy,
         typename std::enable_if<!std::is_integral<IterTy>::value>::type* = nullptr>
std::pair<IterTy, IterTy> block_range(IterTy b, IterTy e, unsigned id, unsigned num) {
  unsigned int dist = std::distance(b, e);
  unsigned int numper = std::max((dist + num - 1) / num, 1U); //round up
  unsigned int A = std::min(numper * id, dist);
  unsigned int B = std::min(numper * (id + 1), dist);
  std::advance(b, A);
  if (dist != B) {
    e = b;
    std::advance(e, B - A);
  }
  return std::make_pair(b,e);
}

template<typename IntTy,
         typename std::enable_if<std::is_integral<IntTy>::value>::type* = nullptr>
std::pair<IntTy, IntTy> block_range(IntTy b, IntTy e, unsigned id, unsigned num) {
  unsigned int dist = e - b;
  unsigned int numper = std::max((dist + num - 1) / num, 1U); //round up
  unsigned int A = std::min(numper * id, dist);
  unsigned int B = std::min(numper * (id + 1), dist);
  b += A;
  if (dist != B) {
    e = b;
    e += (B - A);
  }
  return std::make_pair(b,e);
}

/**
 * Given a prefix sum corresponding to the iterators, divide the iterators
 * up based on the prefix sum.
 *
 * @param edge_prefix_sum Prefix sum of values corresponding to iterators
 * @param begin Beginning of iterator
 * @param end End of iterator
 * @param division_id The division that you want the range for
 * @param num_divisions The total number of divisions you are working with
 * @returns A pair of 2 iterators that correspond to the beginning and the
 * end of the range for the division_id (end not inclusive)
 */
template<typename IterTy>
std::pair<IterTy, IterTy> prefix_range(std::vector<uint64_t> edge_prefix_sum,
                                       IterTy begin, IterTy end, 
                                       uint32_t division_id, 
                                       uint32_t num_divisions) {
  // TODO changed vector uint64_t? 64 bit too big by any chance?
  // TODO change edge var names to something else (since this can be used in
  // a more general sense as well)
  // TODO make accesses to vector more efficient

  assert(division_id < num_divisions);

  uint64_t num_elements = end - begin;
  assert(edge_prefix_sum.size() == num_elements);

  // Single division case
  if (num_divisions == 1) {
    printf("For division %u/%u we have begin %u and end %lu with %lu edges\n", 
           division_id, num_divisions - 1, 
           0, num_elements, edge_prefix_sum.back());
    return std::make_pair(begin, end);
  }

  // Case where we have more divisions than nodes
  if (num_divisions > num_elements) {
    // assign one element per division, i.e. division id n gets assigned to
    // element n (if element n exists, else range is nothing)
    if (division_id < num_elements) {
      IterTy node_to_get = begin + division_id;
      // this division gets a element
      if (division_id == 0) {
        printf("For division %u/%u we have begin %u and end %u with %lu edges\n", 
               division_id, num_divisions - 1, division_id, division_id + 1,
               edge_prefix_sum[0]);
      } else {
        printf("For division %u/%u we have begin %u and end %u with %lu edges\n", 
               division_id, num_divisions - 1, division_id, division_id + 1,
               edge_prefix_sum[division_id] - edge_prefix_sum[division_id - 1]);

      }
      return std::make_pair(node_to_get, node_to_get + 1);
    } else {
      // this division gets no element
      printf("For division %u/%u we have begin %lu and end %lu with 0 edges\n", 
             division_id, num_divisions - 1, num_elements, num_elements);
      return std::make_pair(end, end);
    }
  }

  // To determine range for some element n, you have to determine
  // range for elements 1 through n-1...
  uint32_t current_division = 0;
  uint64_t begin_element = 0;

  uint64_t accounted_edges = 0;
  uint64_t current_element = 0;

  // theoretically how many edges we want to distributed to each division
  uint64_t edges_per_division = edge_prefix_sum.back() / num_divisions;

  //printf("Optimally want %lu edges per division\n", edges_per_division);

  while (current_element < num_elements && current_division < num_divisions) {
    uint32_t divisions_remaining = num_divisions - current_division;

    assert(num_elements - current_element >= divisions_remaining);

    if (divisions_remaining == 1) {
      // assign remaining elements to last division
      assert(current_division == num_divisions - 1);

      //if (current_element != 0) {
      //  printf("For division %u/%u we have begin %lu and end %lu with "
      //         "%lu edges\n", 
      //         division_id, num_divisions - 1, begin_element, 
      //         current_element + 1, 
      //         edge_prefix_sum[current_element] - 
      //           edge_prefix_sum[begin_element - 1]);


      //} else {
      //  printf("For division %u/%u we have begin %lu and end %lu with "
      //         "%lu edges\n", 
      //         division_id, num_divisions - 1, begin_element, 
      //         current_element + 1, edge_prefix_sum.back());
      //}

      return std::make_pair(begin + current_element, end);
    } else if ((num_elements - current_element) == divisions_remaining) {
      // Out of elements to assign: finish up assignments (at this point,
      // each remaining division gets 1 element except for the current
      // division which may have some already)

      for (uint32_t i = 0; i < divisions_remaining; i++) {
        if (current_division == division_id) {
          if (begin_element != 0) {
            printf("For division %u/%u we have begin %lu and end %lu with "
                   "%lu edges\n", 
                   division_id, num_divisions - 1, begin_element, 
                   current_element + 1, 
                   edge_prefix_sum[current_element] - 
                     edge_prefix_sum[begin_element - 1]);

          } else {
            printf("For division %u/%u we have begin %lu and end %lu with "
                   "%lu edges\n", 
                   division_id, num_divisions - 1, begin_element, 
                   current_element + 1, edge_prefix_sum[current_element]);
          }

          return std::make_pair(begin + begin_element, 
                                begin + current_element + 1);
        } else {
          current_division++;
          begin_element = current_element + 1;
          current_element++;
        }
      }

      // shouldn't get out here...
      GALOIS_DIE("should return something before reaching this die statement "
                 "(prefix range)");
    }

    // Determine various edge count numbers
    uint64_t element_edges;
    if (current_element > 0) {
      element_edges = edge_prefix_sum[current_element] - 
                      edge_prefix_sum[current_element - 1];
    } else {
      element_edges = edge_prefix_sum[0];
    }

    uint64_t edge_count_without_current;
    if (current_element > 0) {
      edge_count_without_current = edge_prefix_sum[current_element] -
                                   accounted_edges - element_edges;
    } else {
      edge_count_without_current = 0;
    }
 
    // if this element has a lot of edges, determine if it should go to
    // this division or the next (don't want to cause this division to get
    // too much)
    if (element_edges > (3 * edges_per_division / 4)) {
      // if this current division + edges of this element is too much,
      // then do not add to this division but rather the next one
      if (edge_count_without_current > (edges_per_division / 2)) {

        // finish up this division; its last element is the one before this
        // one
        if (current_division == division_id) {
          printf("For division %u/%u we have begin %lu and end %lu with "
                 "%lu edges\n", division_id, num_divisions - 1, begin_element, 
                 current_element, edge_count_without_current);
  
          return std::make_pair(begin + begin_element, 
                                begin + current_element);
        } else {
          assert(current_division < division_id);

          // this is safe (i.e. won't access -1) as you should never enter this 
          // conditional if current element is still 0
          accounted_edges = edge_prefix_sum[current_element - 1];
          begin_element = current_element;
          current_division++;

          continue;
        }
      }
    }

    // handle this element by adding edges to running sums
    uint64_t edge_count_with_current = edge_count_without_current + 
                                       element_edges;

    if (edge_count_with_current >= edges_per_division) {
      // this division has enough edges after including the current
      // node; finish up

      if (current_division == division_id) {
        printf("For division %u/%u we have begin %lu and end %lu with %lu edges\n", 
               division_id, num_divisions - 1, begin_element, 
               current_element + 1, edge_count_with_current);

        return std::make_pair(begin + begin_element, 
                              begin + current_element + 1);
      } else {
        accounted_edges = edge_prefix_sum[current_element];
        // beginning element of next division
        begin_element = current_element + 1;
        current_division++;
      }
    }

    current_element++;
  }

  // You shouldn't get out here.... (something should be returned before
  // this....)
  GALOIS_DIE("reached end of prefix range when should have returned something");
  return std::make_pair(-1, -1);
}


//! Destroy a range
template<class InputIterator>
void uninitialized_destroy ( InputIterator first, InputIterator last )
{
  typedef typename std::iterator_traits<InputIterator>::value_type T;
  for (; first!=last; ++first)
    (&*first)->~T();
}

}
#endif
