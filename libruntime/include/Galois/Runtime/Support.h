/** Reporting and utility code -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_SUPPORT_H
#define GALOIS_RUNTIME_SUPPORT_H

#include <string>

namespace Galois {

class Statistic;

namespace Runtime {

//! Reports stats for a given thread
void reportStat(const char* loopname, const char* category, unsigned long value);
//! Reports stats for a given thread
void reportStat(const std::string& loopname, const std::string& category, unsigned long value);
//! Reports stats for all threads
void reportStat(Galois::Statistic* value);
//! Reports Galois system memory stats for all threads
void reportPageAlloc(const char* category);
//! Reports NUMA memory stats for all NUMA nodes
void reportNumaAlloc(const char* category);


void reportStatGlobal(const std::string& category, const std::string& val);
void reportStatGlobal(const std::string& category, unsigned long val);


//! Prints all stats
void printStats();

}
} // end namespace Galois

#endif

