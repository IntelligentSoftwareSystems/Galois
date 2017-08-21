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

#ifndef GALOIS_RUNTIME_STATCOLLECTOR_H
#define GALOIS_RUNTIME_STATCOLLECTOR_H

#include "Galois/gdeque.h"
#include "Galois/Substrate/SimpleLock.h"
#include "Galois/Substrate/PerThreadStorage.h"

#include <string>
#include <set>

namespace Galois {
namespace Runtime {

class StatCollector {

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
    char mode; // 0 - int, 1 - double, 2 - string
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
  };

  //stats  HostID,ThreadID,loop,category,instance -> Record
  
  std::map<std::tuple<unsigned,unsigned, const std::string*, const std::string*,unsigned>, RecordTy> Stats;
  Galois::Substrate::SimpleLock StatsLock;

public:


  void addToStat(const std::string& loop, const std::string& category, size_t value, unsigned TID, unsigned HostID);
  void addToStat(const std::string& loop, const std::string& category, double value, unsigned TID, unsigned HostID);
  void addToStat(const std::string& loop, const std::string& category, const std::string& value, unsigned TID, unsigned HostID);

  void printStatsForR(std::ostream& out, bool json);

  //still assumes int values
  void printStats(std::ostream& out);

  void beginLoopInstance(const std::string& str);
};

namespace internal {
  void initStatManager(void);
  void deleteStatManager(void);
}

} // end namespace Runtime
} // end namespace Galois

#endif
