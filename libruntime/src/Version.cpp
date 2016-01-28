/** Implementation for Version Info -*- C++ -*-
 * @file
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Version.h"

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

std::string Galois::getVersion() {
  return STR(GALOIS_VERSION);
}

std::string Galois::getRevision() {
  return "unknown";
}

int Galois::getVersionMajor() {
  return GALOIS_VERSION_MAJOR;
}

int Galois::getVersionMinor() {
  return GALOIS_VERSION_MINOR;
}

int Galois::getVersionPatch() {
  return GALOIS_VERSION_PATCH;
}

int Galois::getCopyrightYear() {
  return GALOIS_COPYRIGHT_YEAR;
}
