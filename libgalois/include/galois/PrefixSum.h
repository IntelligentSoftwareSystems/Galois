#include <cstdint>
#include "galois/WaterFallLock.h"
#include "galois/Galois.h"
#include <variant>

namespace galois {

inline void empty(std::monostate& a, uint64_t i) {
  (void)a;
  (void)i;
}

template <typename T>
inline T equalizer(const T& t) {
  return t;
}

template <typename T>
inline void before(T l, uint64_t tid) {
  l.template wait<1>(tid);
}
template <typename T>
inline void after(T l, uint64_t tid) {
  l.template done<2>(tid);
}

/** This is a struct used for repeated PrefixSums
 * It works using a 2 level algorithm
 * @param A The type of the source array
 * @param B The type of the dst array
 * @param transmute is function A -> B
 * @param scan_op is a function A x B -> B
 * @param combiner is a function B x B -> B
 * @param Conduit is the type used inside the WaterFallLock as well as the Paste
 * array (used for measurement)
 * @param src the source array user is required to ensure the size is correct
 * @param dst the destination array user is required to ensure the size is
 * correct
 * @param lock a reference to a WaterFallLock, which should have length of the
 * number of threads
 * @param paste a conduit assigned per thread in order to ensure cache_line
 * padding for speed
 */
template <typename A, typename B, B (*transmute)(const A&),
          B (*scan_op)(const A& x, const B& y),
          B (*combiner)(const B& x, const B& y),
          template <typename C> typename Conduit>
class PrefixSum {
public:
  /**
   * These are exposed in order to be changed between subsequest calls in the
   * case of dynamic structures
   */
  A* src;
  B* dst;

private:
  using PArr = Conduit<B>;
  Conduit<B> paste;
  using WFLType = galois::WaterFallLock<Conduit<unsigned>>;
  WFLType lock;

  /** Type to make pointers into an array for serial_pfxsum
   *
   */
  template <typename T>
  struct Arr {
    T* arr;

    Arr(T* arr) : arr(arr) {}

    template <typename i_type>
    T& operator[](i_type i) {
      return arr[i];
    }
  };

  /** The templates are used to make this function usable in many different
   * places Enables before and after to take in some context that can be
   * triggered after an object is in the paste array
   * @param Holder is used to specify any object holding A1 (A) or A2 (B)
   */
  template <typename A1, typename A2, A2 (*trans)(const A1&),
            A2 (*scan)(const A1& x, const A2& y), typename CTX,
            void (*before)(CTX&, uint64_t), void (*after)(CTX&, uint64_t),
            template <typename C> typename Holder, bool combine = false>
  inline void serial_pfxsum(Holder<A1> src, Holder<A2> dst, uint64_t ns,
                            CTX ctx) {
    if (!combine)
      dst[0] = trans(src[0]);
    for (uint64_t i = 1; i < ns; i++) {
      before(ctx, i);
      dst[i] = scan(src[i], dst[i - 1]);
      after(ctx, i);
    }
  }

  /** Does the serial pfxsum and puts the final value in the paste_loc for
   * future processing
   *
   */
  inline void parallel_pfxsum_phase_0(A* src, B* dst, uint64_t ns, B& paste_loc,
                                      uint64_t wfl_id) {
    serial_pfxsum<A, B, transmute, scan_op, std::monostate, empty, empty, Arr>(
        src, dst, ns, std::monostate());
    paste_loc = dst[ns - 1];
    lock.template done<1>(wfl_id);
  }

  /** Sums up the paste locations in a single thread to prepare the finished
   * product
   *
   */
  inline void parallel_pfxsum_phase_1(uint64_t ns, uint64_t wfl_id) {
    if (!wfl_id) {
      lock.template done<2>(wfl_id);
      serial_pfxsum<B, B, equalizer, combiner, WFLType&, before<WFLType&>,
                    after<WFLType&>, Conduit>(paste, paste, ns, lock);
    } else {
      lock.template wait<2>(wfl_id - 1);
    }
  }

  /** Does the final prefix sums with the last part of the array being handeled
   * by tid = 0 */
  inline void parallel_pfxsum_phase_2(A* src, B* dst, uint64_t ns,
                                      const B& phase1_val, bool pfxsum) {
    if (pfxsum) {
      dst[0] = scan_op(src[0], phase1_val);
      serial_pfxsum<A, B, transmute, scan_op, std::monostate, empty, empty, Arr,
                    true>(src, dst, ns, std::monostate());
    } else {
      for (uint64_t i = 0; i < ns; i++)
        dst[i] = combiner(phase1_val, dst[i]);
    }
  }

  inline void parallel_pfxsum_work(uint64_t phase0_ind, uint64_t phase0_sz,
                                   uint64_t phase2_ind, uint64_t phase2_sz,
                                   uint64_t wfl_id, uint64_t nt) {

    parallel_pfxsum_phase_0(&src[phase0_ind], &dst[phase0_ind], phase0_sz,
                            paste[wfl_id], wfl_id);

    parallel_pfxsum_phase_1(nt, wfl_id);

    const B& paste_val = paste[wfl_id ? wfl_id - 1 : nt - 1];
    parallel_pfxsum_phase_2(&src[phase2_ind], &dst[phase2_ind], phase2_sz,
                            paste_val, !wfl_id);
  }

  /** This function computes the indices for the different phases and forwards
   * them.
   * @param ns the number of items to sum
   * @param wf_id this corresponds to the thread id
   * @param nt this is the number of threads
   */
  void parallel_pfxsum_op(uint64_t ns, uint64_t wf_id, uint64_t nt) {
    uint64_t div_sz = ns / (nt + 1);
    uint64_t bigs   = ns % (nt + 1);
    uint64_t mid    = nt >> 1;
    bool is_mid     = mid == wf_id;
    // Concentrate the big in the middle thread
    uint64_t phase0_sz = is_mid ? div_sz + bigs : div_sz;
    uint64_t phase0_ind;
    if (wf_id <= mid)
      phase0_ind = div_sz * wf_id;
    else
      phase0_ind = bigs + (div_sz * wf_id);

    uint64_t phase2_sz  = phase0_sz;
    uint64_t phase2_ind = wf_id ? phase0_ind : ns - div_sz;
    parallel_pfxsum_work(phase0_ind, phase0_sz, phase2_ind, phase2_sz, wf_id,
                         nt);
  }

public:
  PrefixSum(A* src, B* dst) : src(src), dst(dst), paste(B()), lock() {}

  /** computePrefixSum is the interface exposed to actually have a prefixSum
   * computed NOTE: this uses on_each be careful!!!
   * @param ns the number of objects in src to sum
   */
  void computePrefixSum(uint64_t ns) {
    galois::on_each([&](unsigned tid, unsigned numThreads) {
      this->parallel_pfxsum_op(ns, tid, numThreads);
    });
    this->lock.reset();
  }

  const char* name() {
    return typeid(PrefixSum<A, B, transmute, scan_op, combiner, Conduit>)
        .name();
  }
};
} // namespace galois
