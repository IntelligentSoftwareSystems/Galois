#ifndef GALOIS_SUBSTRATE_HWTOPO_H
#define GALOIS_SUBSTRATE_HWTOPO_H

#include <vector>

namespace galois {
namespace substrate {

struct threadTopoInfo {
  unsigned tid; // this thread (galois id)
  unsigned socketLeader; //first thread id in tid's package
  unsigned socket; // socket (L3 normally) of thread
  unsigned numaNode; // memory bank.  may be different than socket.
  unsigned cumulativeMaxSocket; // max package id seen from [0, tid]
  unsigned osContext; // OS ID to use for thread binding
  unsigned osNumaNode; // OS ID for numa node
};

struct machineTopoInfo {
  unsigned maxThreads;
  unsigned maxCores;
  unsigned maxPackages;
  unsigned maxNumaNodes;
};

//parse machine topology
std::pair<machineTopoInfo,std::vector<threadTopoInfo>> getHWTopo();
//bind a thread to a hwContext (returned by getHWTopo)
bool bindThreadSelf(unsigned osContext);


} // end namespace substrate
} // end namespace galois

#endif //_HWTOPO_H
