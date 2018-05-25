/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
 */

#ifndef GALOIS_RUNTIME_LOOPSTATISTICS_H
#define GALOIS_RUNTIME_LOOPSTATISTICS_H
namespace galois {
namespace runtime {

// Usually instantiated per thread 
template<bool Enabled>
class LoopStatistics {

protected:
  size_t m_iterations;
  size_t m_pushes;
  size_t m_conflicts;
  const char* loopname;

public:
  explicit LoopStatistics(const char* ln) :
    m_iterations(0), 
    m_pushes(0),
    m_conflicts(0), 
    loopname(ln) { }

  ~LoopStatistics() {
    reportStat_Tsum(loopname, "Iterations", m_iterations);
    reportStat_Tsum(loopname, "Commits", (m_iterations - m_conflicts));
    reportStat_Tsum(loopname, "Pushes", m_pushes);
    reportStat_Tsum(loopname, "Conflicts", m_conflicts);
  }


  size_t iterations(void) const { return m_iterations; }
  size_t pushes(void) const { return m_pushes; }
  size_t conflicts(void) const { return m_conflicts; }

  inline void inc_pushes(size_t v=1) {
    m_pushes += v;
  }

  inline void inc_iterations() {
    ++m_iterations;
  }

  inline void inc_conflicts() {
    ++m_conflicts;
  }
};

template <>
class LoopStatistics<false> {
public:
  explicit LoopStatistics(const char* ln) {}

  size_t iterations(void) const { return 0; }
  size_t pushes(void) const { return 0; }
  size_t conflicts(void) const { return 0; }

  inline void inc_iterations() const { }
  inline void inc_pushes(size_t v=0) const { }
  inline void inc_conflicts() const { }
};

}
}
#endif
