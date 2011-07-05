/*
 * UnorderdMain.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: amber
 */

#include "AVIunordered.h"

int main (int argc, const char* argv[]) {
  AVIunordered* um = new AVIunordered ();
  um->run (argc, argv);
  delete um;
  return 0;
}

