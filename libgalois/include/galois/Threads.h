#ifndef GALOIS_THREADS_H
#define GALOIS_THREADS_H

namespace galois {

/**
 * Sets the number of threads to use when running any Galois iterator. Returns
 * the actual value of threads used, which could be less than the requested
 * value. System behavior is undefined if this function is called during
 * parallel execution or after the first parallel execution.
 */
unsigned int setActiveThreads(unsigned int num) noexcept;

/**
 * Returns the number of threads in use.
 */
unsigned int getActiveThreads() noexcept;

}
#endif
