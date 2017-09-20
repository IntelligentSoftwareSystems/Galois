/** StatCollector -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

// #ifndef GALOIS_RUNTIME_STATCOLLECTOR_H
#if 0
#define GALOIS_RUNTIME_STATCOLLECTOR_H

#include "Galois/gdeque.h"
#include "Galois/gIO.h"
#include "Galois/Substrate/SimpleLock.h"
#include "Galois/Substrate/PerThreadStorage.h"

#include <string>
#include <set>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.

namespace galois {
namespace Runtime {

class StatCollector {

protected:
  template<typename Ty>
  using StringPair = std::pair<const std::string*, Ty>;

  //////////////////////////////////////
  //Symbol Table
  //////////////////////////////////////
  std::set<std::string> symbols;
  const std::string* getSymbol(const std::string& str) const;
  const std::string* getOrInsertSymbol(const std::string& str);

  //////////////////////////////////////
  //Loop instance counter
  //////////////////////////////////////
  std::vector<StringPair<unsigned> > loopInstances;

  unsigned getInstanceNum(const std::string& str) const;
  void addInstanceNum(const std::string& str);

  //////////////////////////////////////
  //Stat list
  //////////////////////////////////////
  struct RecordTy {
    enum Type {
      INT, DOUBLE, STR
    };

    Type mode;

    union {
      size_t valueInt;
      double valueDouble;
      std::string valueStr;
    };

    RecordTy(size_t value);
    RecordTy(double value);
    RecordTy(const std::string& value);
    RecordTy(const RecordTy& r);
    ~RecordTy();

    void print(std::ostream& out) const;

    size_t intVal(void) const;
    double doubleVal(void) const;
    const std::string& strVal(void) const;
  };



  //stats  HostID,ThreadID,loop,category,instance -> Record
  
  std::string m_outfile;
  std::map<std::tuple<unsigned,unsigned, const std::string*, const std::string*,unsigned>, RecordTy> Stats;
  galois::Substrate::SimpleLock StatsLock;

public:

  void printStats(void);

  explicit StatCollector(const std::string& outfile="");

  virtual ~StatCollector(void);

  static boost::uuids::uuid UUID;
  static boost::uuids::uuid getUUID();

  void addToStat(const std::string& loop, const std::string& category, size_t value, unsigned TID, unsigned HostID);
  void addToStat(const std::string& loop, const std::string& category, double value, unsigned TID, unsigned HostID);
  void addToStat(const std::string& loop, const std::string& category, const std::string& value, unsigned TID, unsigned HostID);
  void beginLoopInstance(const std::string& str);

private:
  void printStatsForR(std::ostream& out, bool json);
  void printStats(std::ostream& out);

};


//! Begin a new loop instance
void reportLoopInstance(const char* loopname);
inline void reportLoopInstance(const std::string& loopname) {
  reportLoopInstance(loopname.c_str());
}

//! Reports stats for a given thread
void reportStat(const char* loopname, const char* category, unsigned long value, unsigned TID);
void reportStat(const char* loopname, const char* category, const std::string& value, unsigned TID);
void reportStat(const std::string& loopname, const std::string& category, unsigned long value, unsigned TID);
void reportStat(const std::string& loopname, const std::string& category, const std::string& value, unsigned TID);

void reportStatDist(const std::string& loopname, const std::string& category, const size_t value, unsigned TID, unsigned HostID);
void reportStatDist(const std::string& loopname, const std::string& category, const double value, unsigned TID, unsigned HostID);
void reportStatDist(const std::string& loopname, const std::string& category, const std::string& value, unsigned TID, unsigned HostID);
//! Reports Galois system memory stats for all threads
void reportPageAlloc(const char* category);
//! Reports NUMA memory stats for all NUMA nodes
void reportNumaAlloc(const char* category);

namespace internal {
  void setStatCollector(StatCollector* sc);
}

} // end namespace Runtime
} // end namespace galois

#endif
