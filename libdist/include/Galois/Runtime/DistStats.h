/** DistStatManager: Distributed Statistics Management -*- C++ -*-
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
 * @author M. Amber Hassaan<ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_DIST_STATS_H
#define GALOIS_RUNTIME_DIST_STATS_H

namespace Galois {
namespace Runtime {

class DistStatManager: public Galois::Runtime::StatManager {

  using Base = Galois::Runtime::StatManager;

  using Str = Galois::gstl::String;

  using Base::SEP;

  static constexpr bool PRINT_THREAD_VALS = false; // TODO: use env variable instead of this

  template <typename T>
  struct HostStat {

    using AggStat_ty = typename AggStat<T>::with_mem::with_sum::with_min::with_max;

    using PerHostThreadStats = Galois::gstl::Map<unsigned, AggStat_ty>;
    using InterHostStats = typename AggStat<T>::with_sum::with_min::with_max;

    PerHostThreadStats threadStats;
    InterHostStats interStats;

    void addToStat(unsigned hostID, const Galois::gstl::Vector<T>& threadVals) {

      auto p = threadStats.emplace(hostID, AggStat_ty());
      auto& tstat = p.first->second;

      for (const auto& i: threadVals) {
        tstat.add(i);
        interStats.add(i);
      }

    }

    void print(std::ostream& out) {

      for (const auto& p: threadStats) {
        out << p->first << SEP;

        out << interStats.sum() << SEP << interStats.avg() << SEP << interStats.min() << SEP << interStats.max() << SEP ;
        out << p->second.sum() << SEP << p->second.avg() << SEP << p->second.min() << SEP << p->second.max();

        if (PRINT_THREAD_VALS) {
          for (const T& v: p->second.values()) {
            out << SEP << v;
          }
        }
      }

    }

  };

  template <typename T>
  struct DistStatManagerImpl {

    using StrSet = Galois::gstl::Set<Str>;
    using HostStatMap = Galois::gstl::Map<std::tuple<const Str*, const Str*>, HostStat>;

    StrSet symbols;
    HostStatMap hostStatMap;
    size_t maxThreads = 0;

    const Str* getOrInsertSymbol(const Str& name) {
      auto p = symbols.insert(name);
      return &*(p.first);
    }

    HostStat& getOrInsertMapping(const Str& loopname, const Str& category) {
      const Str* ln = getOrInsertSymbol(loopname);
      const STr* cat = getOrInsertSymbol(category);

      auto tpl = std::make_tuple(ln, cat);

      auto p  = hostStatMap.emplace(tpl, HostStat());

      return p.first->second;
      
    }

    void addToStat(unsigned hostID, const Str& loopname, const Str& category, const Galois::gstl::Vector<T>& threadVals) {

      auto& hstat = getOrInsertMapping(loopname, category);
      hstat.addToStat(hostID, threadVals);

      if (threadVals.size() > maxThreads) {
        maxThreads = threadVals.size();
      }
    }



    void print(std::ostream& out) {
      for (const auto& p: hostStatMap) {
        out << *std::get<0>(p.first) << ", " << *std::get<1>(p.first) << ", ";
        p.print(out);
      }
    }

  };
  
  DistStatManagerImpl<int64_t> intDistStats;
  DistStatManagerImpl<double> fpDistStats;

  void printHeader(std::ostream& out) {
    out << "HOST_ID" << SEP;
    out << "LOOPNAME" << SEP << "CATEGORY" << SEP;
    out << "HOSTS_SUM" << SEP << "HOSTS_AVG" << SEP << "HOSTS_MIN" << SEP << "HOSTS_MAX";


    if (PRINT_THREAD_VALS) {
      out << SEP << "THREAD_SUM" << SEP << "THREAD_AVG" << SEP << "THREAD_MIN" << SEP << "THREAD_MAX";

      for (size_t i = 0; i < intDistStats.maxThreads; ++i) {
        out << SEP << "T" << i;
      }
    }
  }

  virtual void print(std::ostream& out) {
    printHeader(out);

    intDistStats.print(out);
    fpDistStats.print(out);
  }

};

} // end namespace Runtime
} // end namespace Galois


#endif// GALOIS_RUNTIME_DIST_STATS_H
