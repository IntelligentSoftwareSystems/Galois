/** Enviroment Checking Code -*- C++ -*-
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_LL_ENVCHECK_H
#define GALOIS_RUNTIME_LL_ENVCHECK_H

#include <string>

namespace Galois {
namespace Substrate {

//! Return true if the Enviroment variable is set
bool EnvCheck(const char* parm);
bool EnvCheck(const char* parm, int& val);
bool EnvCheck(const char* parm, std::string& val);
bool EnvCheck(const std::string& parm);
bool EnvCheck(const std::string&, int& val);
bool EnvCheck(const std::string&, std::string& val);

} // end namespace Substrate
} // end namespace Galois

#endif
