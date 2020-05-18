#include "galois/runtime/ThreadTimer.h"
#include "galois/runtime/Executor_OnEach.h"
#include "galois/runtime/Statistics.h"

#include <ctime>
#include <limits>

void galois::runtime::ThreadTimers::reportTimes(const char* category,
                                                const char* region) {

  uint64_t minTime = std::numeric_limits<uint64_t>::max();

  for (unsigned i = 0; i < timers_.size(); ++i) {
    auto ns = timers_.getRemote(i)->get_nsec();
    minTime = std::min(minTime, ns);
  }

  std::string timeCat = category + std::string("PerThreadTimes");
  std::string lagCat  = category + std::string("PerThreadLag");

  on_each_gen(
      [&](auto, auto) {
        auto ns  = timers_.getLocal()->get_nsec();
        auto lag = ns - minTime;
        assert(lag > 0 && "negative time lag from min is impossible");

        reportStat_Tmax(region, timeCat.c_str(), ns / 1000000);
        reportStat_Tmax(region, lagCat.c_str(), lag / 1000000);
      },
      std::make_tuple());
}
