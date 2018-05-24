#include "galois/substrate/Barrier.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/gIO.h"

#if defined(GALOIS_HAVE_PTHREAD)

#include <unistd.h>
#include <pthread.h>

#endif


#if defined(GALOIS_HAVE_PTHREAD) && defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS > 0)

namespace {

class PthreadBarrier: public galois::substrate::Barrier {
  pthread_barrier_t bar;

public:
  PthreadBarrier() {
    if (pthread_barrier_init(&bar, 0, ~0))
      GALOIS_DIE("PTHREAD");
  }

  PthreadBarrier(unsigned int v) {
    if (pthread_barrier_init(&bar, 0, v))
      GALOIS_DIE("PTHREAD");
  }

  virtual ~PthreadBarrier() {
    if (pthread_barrier_destroy(&bar))
      GALOIS_DIE("PTHREAD");
  }

  virtual void reinit(unsigned val) {
    if (pthread_barrier_destroy(&bar))
      GALOIS_DIE("PTHREAD");
    if (pthread_barrier_init(&bar, 0, val))
      GALOIS_DIE("PTHREAD");
  }

  virtual void wait() {
    int rc = pthread_barrier_wait(&bar);
    if (rc && rc != PTHREAD_BARRIER_SERIAL_THREAD)
      GALOIS_DIE("PTHREAD");
  }

  virtual const char* name() const { return "PthreadBarrier"; }
};

}

std::unique_ptr<galois::substrate::Barrier> galois::substrate::createPthreadBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(new PthreadBarrier(activeThreads));
}

#else

std::unique_ptr<galois::substrate::Barrier> galois::substrate::createPthreadBarrier(unsigned activeThreads) {
  return std::unique_ptr<Barrier>(nullptr);
}

#endif

