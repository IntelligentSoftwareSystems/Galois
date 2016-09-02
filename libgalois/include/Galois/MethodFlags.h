/** Galois Conflict flags -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_METHODFLAGS_H
#define GALOIS_METHODFLAGS_H

namespace Galois {

/** 
 * What should the runtime do when executing a method.
 *
 * Various methods take an optional parameter indicating what actions
 * the runtime should do on the user's behalf: (1) checking for conflicts,
 * and/or (2) saving undo information. By default, both are performed (ALL).
 */
enum class MethodFlag: char {
  UNPROTECTED = 0,
  WRITE = 1,
  READ = 2,
  INTERNAL_MASK = 3,
  PREVIOUS = 4,
};

//! Bitwise & for method flags
inline MethodFlag operator&(MethodFlag x, MethodFlag y) {
  return (MethodFlag)(((int) x) & ((int) y));
}

//! Bitwise | for method flags
inline MethodFlag operator|(MethodFlag x, MethodFlag y) {
  return (MethodFlag)(((int) x) | ((int) y));
}
}

#endif //GALOIS_METHODFLAGS_H
