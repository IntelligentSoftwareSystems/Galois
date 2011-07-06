/*
 * UnorderdMain.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: amber
 */

#include "AVIunorderedNoLock.h"

int main (int argc, const char* argv[]) {
  AVIunorderedNoLock* um = new AVIunorderedNoLock ();
  um->run (argc, argv);
  delete um;
  return 0;
}

