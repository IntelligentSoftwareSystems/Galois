/** Galois Remote Object Store -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */
#ifndef GALOIS_RUNTIME_COMPILER_HELPER_FUNCTIONS_H
#define GALOIS_RUNTIME_COMPILER_HELPER_FUNCTIONS_H

#include <atomic>
namespace Galois {

  /** Galois::min **/
  template<typename Ty>
    const Ty min(std::atomic<Ty>& a, const Ty& b){
      Ty old_a = a;
      while(a > b){
        a.compare_exchange_strong(old_a, b);
      }
      return a;
    }

  template<typename Ty>
    const Ty& min(Ty& a, const Ty& b) {
      while(a > b){
        a = b;
      }
      return a;
    }

  /** Galois::atomicAdd **/
  template<typename Ty>
    const Ty atomicAdd(std::atomic<Ty>& val, Ty delta){
      Ty old_val;

      do{
        old_val = val;
      }while(!val.compare_exchange_strong(old_val, old_val + delta));

      return old_val;
    }
}
#endif
