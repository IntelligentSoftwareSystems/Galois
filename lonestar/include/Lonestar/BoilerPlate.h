/** Common command line processing for benchmarks -*- C++ -*-
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
 * Common benchmark initialization
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef LONESTAR_BOILERPLATE_H
#define LONESTAR_BOILERPLATE_H

#include "Galois/Galois.h"
#include "Galois/Version.h"
#include "llvm/Support/CommandLine.h"

//! standard global options to the benchmarks
extern llvm::cl::opt<bool> skipVerify;
extern llvm::cl::opt<int> numThreads;
extern llvm::cl::opt<std::string> statFile;


//! initialize lonestar benchmark
void LonestarStart(int argc, char** argv, 
                   const char* app, const char* desc = nullptr, const char* url = nullptr);

#endif
