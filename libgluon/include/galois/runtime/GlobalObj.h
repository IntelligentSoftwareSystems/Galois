/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

/**
 * @file GlobalObj.h
 *
 * Defines the GlobalObject class, which is a base class that other
 * classes inherit from to be assigned a unique global id.
 */

#include <vector>
#include <cstdint>
#include <cassert>

#ifndef _GALOIS_DIST_GLOBAL_OBJECT_H
#define _GALOIS_DIST_GLOBAL_OBJECT_H

namespace galois {
namespace runtime {

/**
 * A class to be inherited from so that all child classes will have a tracked
 * unique ID.
 *
 * @warning Not thread safe: do not concurrently construct GlobalObjects
 */
class GlobalObject {
  //! Vector that points to all GlobalObject instances
  //! @todo make a pointer to avoid static initialization?
  static std::vector<uintptr_t> allobjs;
  //! ID of a global object
  uint32_t objID;

protected:
  GlobalObject(const GlobalObject&) = delete;
  GlobalObject(GlobalObject&&)      = delete;

  /**
   * Returns the pointer for a global object
   *
   * @param oid Global object id to get
   * @returns pointer to requested global object
   */
  static uintptr_t ptrForObj(unsigned oid);

  /**
   * Constructs a global object given a pointer to the object you want to make
   * a global object.
   *
   * @tparam T type of the object to make a GlobalObject
   * @param ptr pointer to object to make a GlobalObject
   *
   * @todo lock needed if multiple GlobalObjects are being constructed in
   * parallel
   */
  template <typename T>
  GlobalObject(const T* ptr) {
    objID = allobjs.size();
    allobjs.push_back(reinterpret_cast<uintptr_t>(ptr));
  }

  /**
   * Returns own global id
   *
   * @returns this object's global id
   */
  uint32_t idForSelf() const { return objID; }
};

} // end namespace runtime
} // end namespace galois

#endif //_GALOIS_DIST_GLOBAL_OBJECT_H
