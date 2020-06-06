/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

//#define _GOPT_DEBUG 1
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>
#include <set>
#include <string>
#include <vector>
#include <libgen.h>

#include <cuda.h>

#include "SGDAsyncEdgeCu.h"

int main(int argc, char ** args) {
   const char * fname = "/net/ohm/export/iss/inputs/GaloisGPU/bgg.gr";
   if (argc == 2)
      fname = args[1];
   typedef SGDAsynEdgeCudaFunctor SGDFunctorTy;
   //fprintf(stderr, "===============================Starting- processing %s\n===============================", fname);
   SGDFunctorTy func(false, fname);
	func(5);
   //fprintf(stderr, "====================Terminating - processed%s================================\n", fname);
   //std::cout << "Completed successfully!\n";
   return 0;
}
