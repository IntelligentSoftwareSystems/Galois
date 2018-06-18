/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_METHODFLAGS_H
#define GALOIS_METHODFLAGS_H

namespace galois {

/**
 * What should the runtime do when executing a method.
 *
 * Various methods take an optional parameter indicating what actions
 * the runtime should do on the user's behalf: (1) checking for conflicts,
 * and/or (2) saving undo information. By default, both are performed (ALL).
 */
enum class MethodFlag : char {
  UNPROTECTED   = 0,
  WRITE         = 1,
  READ          = 2,
  INTERNAL_MASK = 3,
  PREVIOUS      = 4,
};

//! Bitwise & for method flags
inline MethodFlag operator&(MethodFlag x, MethodFlag y) {
  return (MethodFlag)(((int)x) & ((int)y));
}

//! Bitwise | for method flags
inline MethodFlag operator|(MethodFlag x, MethodFlag y) {
  return (MethodFlag)(((int)x) | ((int)y));
}
} // namespace galois

#endif // GALOIS_METHODFLAGS_H
