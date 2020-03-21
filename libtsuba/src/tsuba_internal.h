#ifndef _TSUBA_GLOBAL_DEFS_H_
#define _TSUBA_GLOBAL_DEFS_H_ 1

#include <cerrno>
#include <thread>
#define EXPORT_SYM extern "C" __attribute__((__visibility__("default")))

/* set errno and return */
template <typename T>
static inline T ERRNO_RET(int errno_val, T ret) {
  errno = errno_val;
  return ret;
}

/* get how many threads this machine has */
static inline unsigned n_cpus() {
  auto c = std::thread::hardware_concurrency(); 
  return c != 0 ? c : 1;
}

#endif
