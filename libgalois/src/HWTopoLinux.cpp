/** Machine Descriptions on Linux -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * This contains descriptions of machine topologies.  These describes levels in
 * the machine. SMT Threads &lt; Cores &lt; Packages.
 *
 * This also matches OS cpu numbering to galois thread numbering and binds threads
 * to processors.  Galois threads are assigned densely in each package before the next 
 * package.  SMT threads are bound after all real cores (assuming x86).
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#include "Galois/Substrate/HWTopo.h"
#include "Galois/Substrate/EnvCheck.h"
#include "Galois/gIO.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>

#ifdef GALOIS_USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

#include <sched.h>

using namespace Galois::Substrate;


namespace {

struct cpuinfo {
  // fields filled in from OS files
  unsigned proc;
  unsigned physid;
  unsigned sib;
  unsigned coreid;
  unsigned cpucores;
  unsigned numaNode; //from libnuma
  bool valid; // from cpuset
  bool smt; //computed
};

static bool operator< (const cpuinfo& lhs, const cpuinfo& rhs) {
  if (lhs.smt != rhs.smt)
    return lhs.smt < rhs.smt;
  if (lhs.physid != rhs.physid)
    return lhs.physid < rhs.physid;
  if (lhs.coreid != rhs.coreid)
    return lhs.coreid < rhs.coreid;
  return lhs.proc < rhs.proc;
}

static unsigned getNumaNode(cpuinfo& c) {
  static bool numaAvail = false;
  static bool warnOnce = false;
  if (!warnOnce) {
    warnOnce = true;
#ifdef GALOIS_USE_NUMA
    numaAvail = numa_available() >= 0;
    if (!numaAvail)
      Galois::gWarn("Numa support configured but not present at runtime.  Assuming numa topology matches socket topology.");
#else
    Galois::gWarn("Numa Support Not configured (install libnuma-dev).  Assuming numa topology matches socket topology.");
#endif
  }

#ifdef GALOIS_USE_NUMA
  if (!numaAvail)
    return c.physid;
  int i = numa_node_of_cpu(c.proc);
  if (i < 0)
    GALOIS_SYS_DIE("failed finding numa node for ", c.proc);
  return i;
#else
  return c.physid;
#endif
}

//! Parse /proc/cpuinfo
static std::vector<cpuinfo> parseCPUInfo() {
  std::vector<cpuinfo> vals;
  
  const int len = 1024;
  std::array<char, len> line;
  
  std::ifstream procInfo("/proc/cpuinfo");
  if (!procInfo)
    GALOIS_SYS_DIE("failed opening /proc/cpuinfo");
  
  int cur = -1;
  
  while (true) {
    procInfo.getline(line.data(), len);
    if (!procInfo)
      break;
    
    int num;
    if (sscanf(line.data(), "processor : %d", &num) == 1) {
      assert(cur < num);
      cur = num;
      vals.resize(cur + 1);
      vals.at(cur).proc = num;
    } else if (sscanf(line.data(), "physical id : %d", &num) == 1) {
      vals.at(cur).physid = num;
    } else if (sscanf(line.data(), "siblings : %d", &num) == 1) {
      vals.at(cur).sib = num;
    } else if (sscanf(line.data(), "core id : %d", &num) == 1) {
      vals.at(cur).coreid = num;
    } else if (sscanf(line.data(), "cpu cores : %d", &num) == 1) {
      vals.at(cur).cpucores = num;
    }
  }
  
  for (auto& c : vals)
    c.numaNode = getNumaNode(c);

  return vals;
}

//! Returns physical ids in current cpuset
static std::vector<int> parseCPUSet() {
  std::vector<int> vals;
  
  //Parse: /proc/self/cpuset
  std::string name;
  {
    std::ifstream cpuSetName("/proc/self/cpuset");
    if (!cpuSetName)
      return vals;
    
    //TODO: this will fail to read correctly if name contains newlines
    std::getline(cpuSetName, name);
    if (!cpuSetName)
      return vals;
  }
  
  if (name.size() <= 1)
    return vals;
  
  //Parse: /dev/cpuset/<name>/cpus
  std::string path("/dev/cpuset");
  path += name;
  path += "/cpus";
  std::ifstream cpuSet(path);
  
  if (!cpuSet)
    return vals;
  
  std::string buffer;
  getline(cpuSet,buffer);
  if (!cpuSet)
    return vals;

  size_t current;
  size_t next = -1;
  do {
    current = next + 1;
    next = buffer.find_first_of(',', current );
    auto buf = buffer.substr(current, next - current);
    if (buf.size()) {
      size_t dash = buf.find_first_of('-', 0);
      if (dash != std::string::npos) { //range
        auto first = buf.substr(0, dash);
        auto second = buf.substr(dash+1,std::string::npos);
        unsigned b = atoi(first.data());
        unsigned e = atoi(second.data());
        while (b <= e)
          vals.push_back(b++);
      } else { // singleton
        vals.push_back(atoi(buf.data()));
      }
    }      
  } while (next != std::string::npos);  
  return vals;
}

static unsigned countPackages(const std::vector<cpuinfo>& info) {
  std::set<unsigned> pkgs;
  for (auto& c : info)
    pkgs.insert(c.physid);
  return pkgs.size();
}

static unsigned countCores(const std::vector<cpuinfo>& info) {
  std::set<std::pair<int, int> > cores;
  for (auto& c : info)
    cores.insert(std::make_pair(c.physid, c.coreid));
  return cores.size();
}

static unsigned countNumaNodes(const std::vector<cpuinfo>& info) {
  std::set<unsigned > nodes;
  for (auto& c : info)
    nodes.insert(c.numaNode);
  return nodes.size();
}

static void markSMT(std::vector<cpuinfo>& info) {
  for (unsigned int i = 1 ; i < info.size(); ++i)
    if (info[i - 1].physid == info[i].physid &&
        info[i - 1].coreid == info[i].coreid)
      info[i].smt = true;
    else
      info[i].smt = false;
}

static void markValid(std::vector<cpuinfo>& info) {
  auto v = parseCPUSet();
  if (v.empty()) {
    for (auto& c : info)
      c.valid = true;
  } else {
    std::sort(v.begin(), v.end());
    for (auto& c : info)
      c.valid = std::binary_search(v.begin(), v.end(), c.proc);
  }
}

//FIXME: handle MIC
std::vector<cpuinfo> transform(std::vector<cpuinfo>& info) {
  const bool isMIC = false;
  if (isMIC) {
  }
  return info;
}

} //namespace ""

std::pair<machineTopoInfo, std::vector<threadTopoInfo> > Galois::Substrate::getHWTopo() {
  machineTopoInfo retMTI;

  auto rawInfo = parseCPUInfo();
  std::sort(rawInfo.begin(), rawInfo.end());
  markSMT(rawInfo);
  markValid(rawInfo);

  //Now compute transformed (filtered, reordered, etc) version
  auto info = transform(rawInfo);
  info.erase(std::partition(info.begin(), info.end(), [] (const cpuinfo& c) { return  c.valid; }), info.end());

  std::sort(info.begin(), info.end());
  markSMT(info);
  retMTI.maxPackages = countPackages(info);
  retMTI.maxThreads = info.size();
  retMTI.maxCores = countCores(info);
  retMTI.maxNumaNodes = countNumaNodes(info);

  std::vector<threadTopoInfo> retTTI;
  retTTI.reserve(retMTI.maxThreads);
  //compute renumberings
  std::set<unsigned> sockets;
  std::set<unsigned> numaNodes;
  for (auto& i : info) {
    sockets.insert(i.physid);
    numaNodes.insert(i.numaNode);
  }
  unsigned mid = 0; // max package id
  for (unsigned i = 0; i < info.size(); ++i) {
    unsigned pid = info[i].physid;
    unsigned repid = std::distance(sockets.begin(), sockets.find(info[i].physid));
    mid = std::max(mid, repid);
    unsigned leader = std::distance(info.begin(), std::find_if(info.begin(), info.end(), [pid] (const cpuinfo& c) { return c.physid == pid; }));
    retTTI.push_back(threadTopoInfo{i,
          leader,
          repid,
          (unsigned)std::distance(numaNodes.begin(), numaNodes.find(info[i].numaNode)),
          mid,
          info[i].proc,
          info[i].numaNode
          });
  }

  return std::make_pair(retMTI, retTTI);
} 

//! binds current thread to OS HW context "proc"
bool Galois::Substrate::bindThreadSelf(unsigned osContext) {
#ifndef __CYGWIN__
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO(&mask);
  
  /* CPU_SET sets only the bit corresponding to cpu. */
  // void to cancel unused result warning
  (void)CPU_SET(osContext, &mask);
  
  /* sched_setaffinity returns 0 in success */
  if (sched_setaffinity(0, sizeof(mask), &mask ) == -1) {
    Galois::gWarn("Could not set CPU affinity to ", osContext, "(", strerror(errno), ")");
    return false;
  }
  return true;
#else
  Galois::gWarn("No cpu affinity on Cygwin.  Performance will be bad.");
  return false;
#endif
}
