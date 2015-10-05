/**  -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 */

#include "Galois/DoAllWrap.h"

namespace Galois {

cll::opt<DoAllTypes> doAllKind (
    cll::desc ("DoAll Implementation"),
    cll::values (
      clEnumVal (DOALL_GALOIS, "DOALL_GALOIS"),
      clEnumVal (DOALL_GALOIS_STEAL, "DOALL_GALOIS_STEAL"),
      clEnumVal (DOALL_GALOIS_FOREACH, "DOALL_GALOIS_FOREACH"),
      clEnumVal (DOALL_COUPLED, "DOALL_COUPLED"),
      clEnumVal (DOALL_CILK, "DOALL_CILK"),
      clEnumVal (DOALL_OPENMP, "DOALL_OPENMP"),
      clEnumValEnd),
    cll::init (DOALL_COUPLED));

void setDoAllImpl (const DoAllTypes& type) {
  doAllKind = type;
}

DoAllTypes getDoAllImpl (void) {
  return doAllKind;
}

} // end namespace Galois

