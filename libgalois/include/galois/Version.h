/** Galois Version -*- C++ -*-
 * @file
 * Version of galois accessors
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 *
 */

#ifndef GALOIS_VERSION_H
#define GALOIS_VERSION_H

#include <string>

namespace galois {

std::string getVersion();
std::string getRevision();
int getVersionMajor();
int getVersionMinor();
int getVersionPatch();
int getCopyrightYear();

} // end namespace galois

#endif
