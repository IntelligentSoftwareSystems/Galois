#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <atomic>
#include <sys/syscall.h>
#include <iostream>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <thread>
#include <fcntl.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <cassert>
#include <vector>
#include <ostream>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <time.h>
#include <signal.h>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <stdlib.h>
#include <exception>
#include <algorithm>
#include <climits>
#include <memory>
#include <cstring>
#include <malloc.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/stat.h>
typedef uint32_t uint32;
typedef int32_t int32;
#if !(__APPLE__ & __MACH__)
typedef uint64_t uint64;
typedef int64_t int64;
#else
typedef size_t uint64;
typedef size_t int64;
#endif
typedef uint16_t uint16;
typedef int16_t int16;
typedef int8_t int8;
typedef uint8_t uint8;

#endif /* COMMON_HPP_ */
