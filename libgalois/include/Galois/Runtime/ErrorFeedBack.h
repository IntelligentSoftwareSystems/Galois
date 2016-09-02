/** Error Reporting -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 * Report Errors in a unified ways
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_ERRORFEEDBACK_H
#define GALOIS_RUNTIME_ERRORFEEDBACK_H

namespace Galois {
namespace Runtime {

template<typename... Args>
void gDie(Args&&... args);
template<typename... Args>
void gWarn(Args&&... args);

} // end namespace Runtime
} // end namespace Galois

#endif //_HWTOPO_H
