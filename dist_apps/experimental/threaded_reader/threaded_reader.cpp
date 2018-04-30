/** Threaded Reader -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
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
 *
 * @section Description
 *
 * Reads files with multiple threads.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */
#include <iostream>
#include "galois/gstl.h"
#include "galois/DistGalois.h"
#include "DistBenchStart.h"

constexpr static const char* const regionname = "ThreadedReader";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

static cll::opt<bool> sameFile("sameFile", 
                               cll::desc("All threads read same file"),
                               cll::init(false));

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "Thread Reader";
constexpr static const char* const desc = "Reads files with multiple threads.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  galois::StatTimer readTimer("TIMER_READ", regionname); 

  std::string defaultName = inputFile + "_0.gr";
  std::ifstream fileToRead(defaultName.c_str());
  // get file size
  fileToRead.seekg(0, std::ios_base::end);
  size_t fileSize = fileToRead.tellg();
  fileToRead.seekg(0, std::ios_base::beg);
  fileToRead.close();

  galois::gInfo("File size is ", fileSize);

  // allocate memory for reading
  char* dummyBuffer = (char*)malloc(fileSize);
  char* aOBuffers[galois::getActiveThreads()];

  if (!sameFile) {
    for (unsigned i = 0; i < galois::getActiveThreads(); i++) {
      aOBuffers[i] = (char*)malloc(fileSize);
    } 
  } else {
    dummyBuffer = (char*)malloc(fileSize);
  }

  galois::gInfo("Beginning file read");

  readTimer.start();
  galois::on_each(
    [&] (unsigned tid, unsigned numThreads) {
      // each reads different file (should have same size)
      if (!sameFile) {
        std::string localName = inputFile + "_" + std::to_string(tid) + ".gr";
        galois::gInfo("[", tid, "] Reading ", localName);
        std::ifstream myLocalFile(localName);

        size_t numBytesToLoad = fileSize;
        size_t bytesRead = 0;

        while (numBytesToLoad > 0) {
          myLocalFile.read(((char*)aOBuffers[tid]) + bytesRead, 
                           numBytesToLoad);
          size_t numRead = myLocalFile.gcount();
          numBytesToLoad -= numRead;
          bytesRead += numRead;
        }

        myLocalFile.close();
      // each reads same file
      } else {
        // get my own range
        auto myRange = galois::block_range((size_t)0, fileSize, tid, numThreads);
        size_t myBegin = myRange.first;

        galois::gInfo("[", tid, "] Reading same ", defaultName);
        // open up local file and jump to this thread's assigned beginning
        std::ifstream myLocalFile(defaultName);
        myLocalFile.seekg(myBegin, std::ios_base::beg);

        size_t myEnd = myRange.second;
        size_t numBytesToLoad = myEnd - myBegin;
        size_t bytesRead = 0;
    
        while (numBytesToLoad > 0) {
          myLocalFile.read(((char*)dummyBuffer) + myBegin + bytesRead, 
                           numBytesToLoad);
          size_t numRead = myLocalFile.gcount();
          numBytesToLoad -= numRead;
          bytesRead += numRead;
        }

        myLocalFile.close();
      }
    }
  );
  readTimer.stop();

  size_t totalRead = fileSize;

  // free up allocated memory
  if (!sameFile) {
    for (unsigned i = 0; i < galois::getActiveThreads(); i++) {
      free(aOBuffers[i]);
    } 
    totalRead *= galois::getActiveThreads();
  } else {
    free(dummyBuffer);
  }

  galois::gInfo("Total read is ", totalRead);

  galois::gInfo("[", galois::runtime::getSystemNetworkInterface().ID, "] Time "
                "in seconds is ", readTimer.get() / 1000.0, " (", 
                (float)totalRead / readTimer.get() / 1000.0, " MBPS)");

  return 0;
}
