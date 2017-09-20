/** Iterator Range helper -*- C++ -*-
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Wraps Author O'Dwyer's public domain code.  As noted below, those
 * specific codes are not under the LGPL.
 */

#ifndef GALOIS_RUNTIME_ITERABLE_H
#define GALOIS_RUNTIME_ITERABLE_H

namespace galois {
namespace Runtime {

//iterable and make_iterable specific
//From: https://github.com/CppCon/CppCon2014/tree/master/Presentations/C%2B%2B11%20in%20the%20Wild%20-%20Techniques%20from%20a%20Real%20Codebase
//Author: Arthur O'Dwyer
//License: The C++ code in this directory is placed in the public domain and may be reused or modified for any purpose, commercial or non-commercial.

template<class It>
class iterable
{
  It m_first, m_last;
public:
  iterable() = default;
  iterable(It first, It last) :
    m_first(first), m_last(last) {}
  It begin() const { return m_first; }
  It end() const { return m_last; }
};

template<class It>
static inline iterable<It> make_iterable(It a, It b)
{
  return iterable<It>(a, b);
}

} // end namespace Runtime
} // end namespace galois

#endif //GALOIS_RUNTIME_ITERABLE_H
