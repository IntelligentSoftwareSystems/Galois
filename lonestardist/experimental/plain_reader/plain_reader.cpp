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

#include <iostream>
#include "galois/gstl.h"
#include "galois/DistGalois.h"
#include "DistBenchStart.h"

constexpr static const char* const regionname = "PlainReader";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

static cll::opt<bool> sameFile("sameFile",
                               cll::desc("All processes read same file"),
                               cll::init(false));

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "Plain Reader";
constexpr static const char* const desc = "Reads files depending on host ID.";
constexpr static const char* const url  = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  galois::StatTimer readTimer("TIMER_READ", regionname);

  std::string netNum =
      std::to_string(galois::runtime::getSystemNetworkInterface().ID);

  // if same file is set make all processes read _0
  if (sameFile) {
    netNum = "0";
  }

  std::string newName = inputFile + "_" + netNum + ".gr";
  galois::gInfo("[", galois::runtime::getSystemNetworkInterface().ID,
                "] Reading ", newName);

  std::ifstream fileToRead(newName.c_str());

  // get file size
  fileToRead.seekg(0, std::ios_base::end);
  size_t fileSize = fileToRead.tellg();
  fileToRead.seekg(0, std::ios_base::beg);

  char* dummyBuffer = (char*)malloc(fileSize);

  size_t bytesRead      = 0;
  size_t numBytesToLoad = fileSize;
  uint64_t numLoops     = 0;

  // begin read
  readTimer.start();

  while (numBytesToLoad > 0) {
    fileToRead.read(((char*)dummyBuffer) + bytesRead, numBytesToLoad);
    size_t numRead = fileToRead.gcount();
    numBytesToLoad -= numRead;
    bytesRead += numRead;
    numLoops++;
  }
  readTimer.stop();

  fileToRead.close();
  free(dummyBuffer);
  galois::gInfo("[", galois::runtime::getSystemNetworkInterface().ID,
                "] Num "
                "read ops called is ",
                numLoops);
  galois::gInfo("[", galois::runtime::getSystemNetworkInterface().ID,
                "] Time "
                "in seconds is ",
                readTimer.get() / 1000.0, " (",
                (float)bytesRead / readTimer.get() / 1000.0, " MBPS)");
  return 0;
}
