#ifndef PERFCOUNTERS_HPP
#define PERFCOUNTERS_HPP

#include "papi.h"

class PerfStatistics {
private:
    long long flops = 0;
    int eventSet = PAPI_NULL;
public:
    int start() {
        retval = PAPI_library_init(PAPI_VER_CURRENT);
        if (retval < 0) {
            return retval;
        }

        PAPI_create_eventset(&eventSet);

        if (PAPI_add_event(eventSet, PAPI_FP_OPS) != PAPI_OK) {
            return -1;
        }

        if (PAPI_start(eventSet) != PAPI_OK) {
            return -1;
        }

        return 0;
    }

    int stop() {
        if (PAPI_read(EventSet, &flops) != PAPI_OK) {
            return -1;
        }
    }

    unsigned long getFLOPs();

};

#endif // PERFCOUNTERS_HPP
