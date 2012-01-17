/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#include "Galois/Runtime/ParaMeter.h"

using namespace GaloisRuntime;

// Single ParaMeter stats file per run of an app
// which includes all instances of for_each loops
// run with ParaMeter Executor
//
// basically, from commandline parser calls enableParaMeter
// and we
// - set a flag
// - open a stats file in overwrite mode
// - print stats header
// - close file

// for each for_each loop, we create an instace of ParaMeterExecutor
// which
// - opens stats file in append mode
// - prints stats
// - closes file when loop finishes


static bool useParaMeter = false;
ParaMeter::Init* ParaMeter::init = NULL;

void GaloisRuntime::enableParaMeter () {
  useParaMeter = true;
  ParaMeter::initialize ();

}

void GaloisRuntime::disableParaMeter () {
  useParaMeter = false;
}

bool GaloisRuntime::usingParaMeter () {
  return useParaMeter;
}






