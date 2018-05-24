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
