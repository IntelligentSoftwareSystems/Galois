/** Distributed Galois System Interface -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 */

#ifndef GALOIS_DIST_GALOIS_H
#define GALOIS_DIST_GALOIS_H

#include "galois/runtime/Init.h"
#include "galois/runtime/DistStats.h"

#include <string>
#include <utility>
#include <tuple>

/**
 * Main Galois namespace. All the core Galois functionality will be found in here.
 */
namespace galois {

/**
 * Explicit class to initialize the Galois Runtime
 * Runtime is destroyed when this object is destroyed
 */
class DistMemSys: public runtime::SharedMemRuntime<runtime::DistStatManager> {

public:
  explicit DistMemSys(void);

  ~DistMemSys(void);
};

} // namespace galois
#endif
