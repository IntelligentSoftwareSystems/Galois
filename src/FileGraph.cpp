#include "Galois/Graphs/FileGraph.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>

using namespace Galois::Graph;

//File format V1:
//all values are uint64_t LE
//version (1)
//numNodes
//numEdges
//outindexs (outindex[nodeid] is index of first edge for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
//outedges


FileGraph::FileGraph()
  : masterMapping(0), masterLength(0), masterFD(0),
    outIdx(0), outs(0), 
    numEdges(0), numNodes(0)
{
}

FileGraph::~FileGraph() {
  if (masterMapping)
    munmap(masterMapping, masterLength);
  if (masterFD)
    close(masterFD);
    
}

bool FileGraph::fromFile(const char* filename) {
  masterFD = open(filename, O_RDONLY);
  
  struct stat buf;
  int f = fstat(masterFD, &buf);
  masterLength = buf.st_size;

  void* m = mmap(0, masterLength, PROT_READ,MAP_PRIVATE, masterFD, 0);
  if (m == MAP_FAILED) {
    m = 0;
    return false;
  }
  masterMapping = m;

  //parse file
  uint64_t* fptr = (uint64_t*)m;
  uint64_t version = *fptr++;
  assert(version == 1);
  numNodes = *fptr++;
  numEdges = *fptr++;
  outIdx = fptr;
  outs = fptr + numNodes;
}
