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
  using ThreadVals = Galois::gstl::Vector<T>;

  template <typename T>
  using HostStatVal = std::pair<unsigned, const ThreadVals<T>&>;

  template <typename T>
  struct HostStat {

    using AggStat_ty = typename AggStat<T>::with_mem::with_sum::with_min::with_max;

    using PerHostThreadStats = Galois::gstl::Map<unsigned, AggStat_ty>;
    using InterHostStats = typename AggStat<T>::with_sum::with_min::with_max;

    PerHostThreadStats threadStats;
    InterHostStats hostStats;

    void add(const HostStatVal<T>& val) {

      const auto& hostID = val.first;
      const auto& threadVals = val.second;

      auto p = threadStats.emplace(hostID, AggStat_ty());
      auto& tstat = p.first->second;

      for (const auto& i: threadVals) {
        tstat.add(i);
        hostStats.add(i);
      }

    }

    void print(std::ostream& out) {

      for (const auto& p: threadStats) {
        out << p->first << SEP;

        out << hostStats.sum() << SEP << hostStats.avg() << SEP << hostStats.min() << SEP << hostStats.max() << SEP ;
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
  struct DistStatManagerImpl: public hidden::BasicStatMap<HostStatVal<T> > {

    using Base = hidden::BasicStatMap<HostStatVal<T> >;

    void print(std::ostream& out) {

      for (auto i = Base::cbegin(), end_i = cend(); i != end_i; ++i) {

        out << Base::name(i) << SEP << Base::category(i) << SEP;
        p.print(out);
        out << std::endl;
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

      for (unsigned i = 0; i < Base::maxThreads(); ++i) {
        out << SEP << "T" << i;
      }
    }

    out << std::endl;
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
