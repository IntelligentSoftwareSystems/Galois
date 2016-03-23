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
#include <algorithm>
#include <vector>

namespace Galois {
  /** Galois::min **/
  template<typename Ty>
    const Ty atomicMin(std::atomic<Ty>& a, const Ty& b){
      Ty old_a = a;
      //std::cout << " b : " << b <<"\n";
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

  /** Pair Wise Average function **/
  template<typename Ty>
  const Ty pairWiseAvg(Ty a, Ty b) {
    return (a+b)/2.0;
  }

  template<typename Ty>
  void pairWiseAvg_vec(std::vector<Ty>& a_vec, std::vector<Ty>& b_vec) {
    for(unsigned i = 0; i < a_vec.size(); ++i) {
      a_vec[i] = (a_vec[i] + b_vec[i])/2.0;
    }
  }

  template<typename Ty>
  void resetVec(std::vector<Ty>& a_vec) {
    std::for_each(a_vec.begin(), a_vec.end(),[](Ty &ele){ele = 0;} ); 
  }

  //like std::inner_product
  template<typename ItrTy, typename Ty >
  Ty innerProduct(ItrTy a_begin, ItrTy a_end, ItrTy b_begin, Ty init_value) {
    auto jj = b_begin;
    for(auto ii = a_begin; ii != a_end; ++ii, ++jj){
      init_value += (*ii) * (*jj);
    }
    return init_value;
  }
}//End namespace Galois
#endif
