/** Galois user interface -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_THREADS_H
#define GALOIS_THREADS_H

namespace Galois {

/**
 * Sets the number of threads to use when running any Galois iterator. Returns
 * the actual value of threads used, which could be less than the requested
 * value. System behavior is undefined if this function is called during
 * parallel execution or after the first parallel execution.
 */
unsigned int setActiveThreads(unsigned int num) noexcept;

/**
 * Returns the number of threads in use.
 */
unsigned int getActiveThreads() noexcept;

}
#endif
