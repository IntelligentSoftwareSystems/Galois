// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// dmp.h: The DMP interface exposed to user code
//

#ifndef _DMP_H_
#define _DMP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "dmp-common.h"

int main(int argc, char* argv[]);
#define main DMPmain

#define pthread_create(a, b, c, d) DMPthread_create(a, b, c, d)
#define pthread_join(a, b) DMPthread_join(a, b)
#define pthread_cancel(a) DMPthread_cancel(a)

#define pthread_once_t DMPonce
#undef PTHREAD_ONCE_INITIALIZER
#define PTHREAD_ONCE_INITIALIZER DMP_THREAD_ONCE_INITIALIZER
#define pthread_once(a, b) DMPonce_once(a, b)

#define pthread_mutex_t DMPmutex
#undef PTHREAD_MUTEX_INITIALIZER
#define PTHREAD_MUTEX_INITIALIZER DMP_THREAD_MUTEX_INITIALIZER
#define pthread_mutex_init(a, b) DMPmutex_init(a, b)
#define pthread_mutex_destroy(a) DMPmutex_destroy(a)
#define pthread_mutex_lock(a) DMPmutex_lock(a)
#define pthread_mutex_trylock(a) DMPmutex_trylock(a)
#define pthread_mutex_unlock(a) DMPmutex_unlock(a)

#define pthread_rwlock_t DMPrwlock
#undef PTHREAD_RWLOCK_INITIALIZER
#define PTHREAD_RWLOCK_INITIALIZER DMP_THREAD_RWLOCK_INITIALIZER
#define pthread_rwlock_init(a, b) DMPrwlock_init(a, b)
#define pthread_rwlock_destroy(a) DMPrwlock_destroy(a)
#define pthread_rwlock_rdlock(a) DMPrwlock_rdlock(a)
#define pthread_rwlock_wrlock(a) DMPrwlock_wrlock(a)
#define pthread_rwlock_tryrdlock(a) DMPrwlock_tryrdlock(a)
#define pthread_rwlock_trywrlock(a) DMPrwlock_trywrlock(a)
#define pthread_rwlock_unlock(a) DMPrwlock_unlock(a)

#define pthread_cond_t DMPcondvar
#undef PTHREAD_COND_INITIALIZER
#define PTHREAD_COND_INITIALIZER DMP_THREAD_COND_INITIALIZER
#define pthread_cond_init(a, b) DMPcondvar_init(a, b)
#define pthread_cond_destroy(a) DMPcondvar_destroy(a)
#define pthread_cond_signal(a) DMPcondvar_signal(a)
#define pthread_cond_wait(a, b) DMPcondvar_wait(a, b)
#define pthread_cond_broadcast(a) DMPcondvar_broadcast(a)
#define pthread_cond_timedwait(a, b)                                           \
  (error_pthread_cond_timedwait_not_implemented())

#define pthread_barrier_t DMPbarrier
#define pthread_barrier_init(a, b, c) DMPbarrier_init(a, b, c)
#define pthread_barrier_destroy(a) DMPbarrier_destroy(a)
#define pthread_barrier_wait(a) DMPbarrier_wait(a)

// these attributes are not implemented
#define pthread_barrierattr_init(a)                                            \
  (error_pthread_barrierattr_init_not_implemented())
#define pthread_barrierattr_destroy(a)                                         \
  (error_pthread_barrierattr_destroy_not_implemented())
#define pthread_condattr_init(a) (error_pthread_condattr_init_not_implemented())
#define pthread_condattr_destroy(a)                                            \
  (error_pthread_condattr_destroy_not_implemented())
#define pthread_mutexattr_init(a)                                              \
  (error_pthread_mutexattr_init_not_implemented())
#define pthread_mutexattr_destroy(a)                                           \
  (error_pthread_mutexattr_destroy_not_implemented())
#define pthread_rwlockattr_init(a)                                             \
  (error_pthread_rwlockattrs_init_not_implemented())
#define pthread_rwlockattr_destroy(a)                                          \
  (error_pthread_rwlockattr_destroy_not_implemented())

//--------------------------------------------------------------
// HACK for C++: see include/c++/.../bits/gthr-default.h
//--------------------------------------------------------------

#undef _GLIBCXX_GTHREAD_USE_WEAK
#define _GLIBCXX_GTHREAD_USE_WEAK 0

#ifdef __cplusplus
}
#endif

#endif // _DMP_H_
