/** Common command line processing for benchmarks -*- C++ -*-
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
 * @section Description
 *
 * Perform common setup tasks for the lonestar benchmarks
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Lonestar/BoilerPlate.h"
#include "galois/Version.h"
#include "galois/runtime/Statistics.h"

#include <iostream>
#include <sstream>

//! standard global options to the benchmarks
llvm::cl::opt<bool> skipVerify("noverify", llvm::cl::desc("Skip verification step"), llvm::cl::init(false));
llvm::cl::opt<int> numThreads("t", llvm::cl::desc("Number of threads"), llvm::cl::init(1));
llvm::cl::opt<std::string> statFile("statFile", llvm::cl::desc("ouput file to print stats to "), llvm::cl::init(""));


static void LonestarPrintVersion() {
  std::cout << "Galois Benchmark Suite v" << galois::getVersion() << " (" << galois::getRevision() << ")\n";
}


//! initialize lonestar benchmark
void LonestarStart(int argc, char** argv, 
    const char* app, const char* desc, const char* url) {

  llvm::cl::SetVersionPrinter(LonestarPrintVersion);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  numThreads = galois::setActiveThreads(numThreads); 

  galois::runtime::setStatFile(statFile);

  LonestarPrintVersion();
  std::cout << "Copyright (C) " << galois::getCopyrightYear() << " The University of Texas at Austin\n";
  std::cout << "http://iss.ices.utexas.edu/galois/\n\n";
  std::cout << "application: " <<  (app ? app : "unspecified") << "\n";
  if (desc)
    std::cout << desc << "\n";
  if (url)
    std::cout << "http://iss.ices.utexas.edu/?p=projects/galois/benchmarks/" << url << "\n";
  std::cout << "\n";

  std::ostringstream cmdout;
  for (int i = 0; i < argc; ++i) {
    cmdout << argv[i];
    if (i != argc - 1)
      cmdout << " ";
  }
  galois::runtime::reportParam("(NULL)", "CommandLine", cmdout.str());
  galois::runtime::reportParam("(NULL)", "Threads", numThreads);

  char name[256];
  gethostname(name, 256);
  galois::runtime::reportParam("(NULL)", "Hostname", name);

}
