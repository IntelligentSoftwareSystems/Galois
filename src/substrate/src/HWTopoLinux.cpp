/** Machine Descriptions on Linux -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Substrate/HWTopo.h"
#include "Galois/Substrate/EnvCheck.h"
#include "Galois/Substrate/gio.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <functional>
#include <vector>
#include <set>

#include <sched.h>

using namespace Galois::Substrate;

namespace {

static const char* sProcInfo = "/proc/cpuinfo";
static const char* sCPUSet   = "/proc/self/cpuset";

struct cpuinfo {
  // fields filled in from OS files
  unsigned proc;
  unsigned physid;
  unsigned sib;
  unsigned coreid;
  unsigned cpucores;
  bool valid; // from cpuset
  bool smt;
};

bool operator< (const cpuinfo& lhs, const cpuinfo& rhs) {
  if (lhs.smt != rhs.smt)
    return lhs.smt < rhs.smt;
  if (lhs.physid != rhs.physid)
    return lhs.physid < rhs.physid;
  if (lhs.coreid != rhs.coreid)
    return lhs.coreid < rhs.coreid;
  return lhs.proc < rhs.proc;
}

//! binds current thread to OS HW context "proc"
static bool bindToProcessor(unsigned proc) {
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO(&mask);
  
  /* CPU_SET sets only the bit corresponding to cpu. */
  // void to cancel unused result warning
  (void)CPU_SET(proc, &mask);
  
  /* sched_setaffinity returns 0 in success */
  if (sched_setaffinity(0, sizeof(mask), &mask ) == -1) {
    gWarn("Could not set CPU affinity for thread ", proc, "(", strerror(errno), ")");
    return false;
  }
  return true;
}

//! Parse /proc/cpuinfo
static std::vector<cpuinfo> parseCPUInfo() {
  std::vector<cpuinfo> vals;
  
  const int len = 1024;
  std::array<char, len> line;
  
  std::ifstream procInfo(sProcInfo);
  if (!procInfo)
    GALOIS_SYS_DIE("failed opening ", sProcInfo);
  
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
  
  return vals;
}

//! Returns physical ids in current cpuset
static std::vector<int> parseCPUSet() {
  std::vector<int> vals;
  
  //Parse: /proc/self/cpuset
  std::string name;
  {
    std::ifstream cpuSetName(sCPUSet);
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
  
  const char* np = buffer.data();
  while (np && strlen(np)) {
    char* c = index(np, ',');  
    if (c) { //slice string at comma (np is old string, c is next string)
      *c = '\0';
      ++c;
    }
    
    char* d = index(np, '-');
    if (d) { //range
      *d = '\0';
      ++d;
      unsigned b = atoi(np);
      unsigned e = atoi(d);
      while (b <= e)
        vals.push_back(b++);
    } else { //singleton
      vals.push_back(atoi(np));
    }
    np = c;
  }
  
  return vals;
}

class HWTopoLinux : public HWTopo {

  //number of hw supported threads
  unsigned numThreads;
  
  //number of "real" processors
  unsigned numCores;

  //number of packages
  unsigned numPackages;

  std::vector<threadInfo> tInfo;

  //  std::vector<int> packages;
  //  std::vector<int> maxPackage;
  //  std::vector<int> virtmap;
  //  std::vector<int> leaders;

  unsigned countPackages(const std::vector<cpuinfo>& info) const {
    std::set<unsigned> pkgs;
    for (auto& c : info)
      pkgs.insert(c.physid);
    return pkgs.size();
  }

  unsigned countCores(const std::vector<cpuinfo>& info) const {
    std::set<std::pair<int, int> > cores;
    for (auto& c : info)
      cores.insert(std::make_pair(c.physid, c.coreid));
    return cores.size();
  }

  void markSMT(std::vector<cpuinfo>& info) {
    for (int i = 1 ; i < info.size(); ++i)
      if (info[i - 1].physid == info[i].physid &&
          info[i - 1].coreid == info[i].coreid)
        info[i].smt = true;
      else
        info[i].smt = false;
  }


  void markValid(std::vector<cpuinfo>& info) {
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

  void renumber(std::vector<cpuinfo>& info) {
    std::set<unsigned> pkgNum;
    for (auto& c : info)
      pkgNum.insert(c.physid);
    for (auto& c : info)
      c.physid = std::distance(pkgNum.begin(), pkgNum.find(c.physid));
  }

  //FIXME: handle MIC
  std::vector<cpuinfo> transform(std::vector<cpuinfo>& info) {
    const bool isMIC = false;
    if (isMIC) {
    }
    return info;
  }

public:
  HWTopoLinux() {
    std::vector<cpuinfo> rawInfo = parseCPUInfo();
    std::sort(rawInfo.begin(), rawInfo.end());
    markSMT(rawInfo);
    markValid(rawInfo);

    //Now compute transformed (filtered, reordered, etc) version
    auto info = transform(rawInfo);
    //filter invalid
    info.erase(std::partition(info.begin(), info.end(), [] (const cpuinfo& c) { return  c.valid; }), info.end());
    //densly number packages
    std::sort(info.begin(), info.end());
    markSMT(info);
    renumber(info);

    numThreads = info.size();
    numCores = countCores(info);
    numPackages = countPackages(info);

    tInfo.resize(numThreads);
    for (unsigned i = 0; i < numThreads; ++i) {
      tInfo[i].hwContext = info[i].proc;
      tInfo[i].package = info[i].physid;
      tInfo[i].isPackageLeader = false;
    }
    //.isPackageLeader
    //.packageLeader
  }

  virtual const threadInfo& getThreadInfo(unsigned tid) const {    
    return tInfo[tid];
  }

  virtual bool bindThreadToProcessor(unsigned galois_thread_id) const {
    return bindToProcessor(getThreadInfo(galois_thread_id).hwContext);
  }

  virtual unsigned getMaxThreads() const {
    return numThreads;
  }

  virtual unsigned getMaxCores() const {
    return numCores;
  }

  virtual unsigned getMaxPackages() const {
    return numPackages;
  }

};

} //namespace

std::unique_ptr<Galois::Substrate::HWTopo> Galois::Substrate::getHWTopo() {
  return std::unique_ptr<Galois::Substrate::HWTopo>(new HWTopoLinux());
}
