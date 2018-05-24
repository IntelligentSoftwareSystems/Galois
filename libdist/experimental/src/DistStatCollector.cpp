#if 0 // disabled
#include "galois/runtime/DistStatCollector.h"

#include <cmath>
#include <new>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace galois;
using namespace galois::runtime;

DistStatCollector::DistStatCollector(const std::string& outfile): StatCollector(outfile) {}


static void recvAtHost_0_int(uint32_t HostID, const std::string loopname, const std::string category, const size_t value, unsigned TID) {
  reportStatDist(loopname, category, value, TID, HostID);
}

static void recvAtHost_0_double(uint32_t HostID, const std::string loopname, const std::string category, const double value, unsigned TID) {
  reportStatDist(loopname, category, value, TID, HostID);
}

static void recvAtHost_0_str(uint32_t HostID, const std::string loopname, const std::string category, const std::string value, unsigned TID) {
  reportStatDist(loopname, category, value, TID, HostID);
}

void DistStatCollector::combineAtHost_0(void) {
  galois::runtime::getHostBarrier().wait();

  if (getHostID() != 0) {
    for (auto& p : Stats) {
      const auto tid = std::get<1>(p.first);
      const auto& loopname = *std::get<2>(p.first);
      const auto& category = *std::get<3>(p.first);


      switch (p.second.mode) {

        case StatCollector::RecordTy::INT: 
          {
            size_t val = p.second.intVal();
            getSystemNetworkInterface().sendSimple(0, recvAtHost_0_int, loopname, category, val, tid);
            break;
          }

        case StatCollector::RecordTy::DOUBLE: 
          {
            double val = p.second.doubleVal();
            getSystemNetworkInterface().sendSimple(0, recvAtHost_0_double, loopname, category, val, tid);
            break;
          }

        case StatCollector::RecordTy::STR: 
          {
            const std::string& val = p.second.strVal();
            getSystemNetworkInterface().sendSimple(0, recvAtHost_0_str, loopname, category, val, tid);
            break;
          }
        default:
          std::abort();
          break;
      }

    }
  }

  galois::runtime::getHostBarrier().wait();
}


void DistStatCollector::printStats(void) {

  combineAtHost_0();

  if (getHostID() != 0)  {
    return;

  } else {
    StatCollector::printStats();
  }

}

#endif
