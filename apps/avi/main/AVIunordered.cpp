/*
 * AVIunordered.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: amber
 */

#include "AVIunordered.h"

int main (int argc, char* argv[]) {
  Galois::StatManager M;
  AVIunordered* um = new AVIunordered ();
  um->run (argc, argv);
  delete um;
  return 0;
}

