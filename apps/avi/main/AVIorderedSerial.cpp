/*
 * AVIorderedSerial.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: amber
 */
#include "AVIabstractMain.h"

int main (int argc, char* argv[]) {
  AVIorderedSerial* serial = new AVIorderedSerial ();
  serial->run (argc, argv);
  delete serial;
  return 0;
}

