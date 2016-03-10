/** SSSP -*- C++ -*-
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
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 */
#include <vector>
#include <cstdint>

#ifndef _GALOIS_DIST_GLOBAL_OBJECT_H
#define _GALOIS_DIST_GLOBAL_OBJECT_H
class GlobalObject {
  //FIXME: lock?
//  static std::vector<uintptr_t> allobjs;
  uint32_t objID;

 protected:
  static uintptr_t ptrForObj(unsigned oid);

  GlobalObject(const GlobalObject&) = delete;
  GlobalObject(GlobalObject&&) = delete;

  /*
   * Replace a global var with a static. No longer needs a separate cpp file.
   * */
  static std::vector<uintptr_t> & allobjs(){
     static std::vector<uintptr_t> all_objects;
     return all_objects;
  }

  template<typename T>
  GlobalObject(const T* ptr) {
    objID = GlobalObject::allobjs().size();
    GlobalObject::allobjs().push_back(reinterpret_cast<uintptr_t>(ptr));
  }

  uint32_t idForSelf() const {
    return objID;
  }
};


#endif//_GALOIS_DIST_GLOBAL_OBJECT_H
