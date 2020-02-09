#include <iomanip>
#include <sys/time.h>
#include <sys/resource.h>
#include "gnn.h"
#include "math_functions.hpp"

class ResourceManager {
public:
	ResourceManager() {}
	~ResourceManager(){}
	//peak memory usage
	std::string get_peak_memory() {
		double kbm;
		struct rusage CurUsage;
		getrusage(RUSAGE_SELF, &CurUsage);
		kbm = (double)CurUsage.ru_maxrss;
		double mbm = kbm / 1024.0;
		double gbm = mbm / 1024.0;
		return
			"Peak memory: " +
			to_string_with_precision(mbm, 3) + " MB; " +
			to_string_with_precision(gbm, 3) + " GB";
	}
private:
	template <typename T = double>
	std::string to_string_with_precision(const T a_value, const int& n) {
		std::ostringstream out;
		out << std::fixed;
		out << std::setprecision(n) << a_value;
		return out.str();
	}
};

class Timer {
public:
	Timer() {}
	void Start() { gettimeofday(&start_time_, NULL); }
	void Stop() {
		gettimeofday(&elapsed_time_, NULL);
		elapsed_time_.tv_sec  -= start_time_.tv_sec;
		elapsed_time_.tv_usec -= start_time_.tv_usec;
	}
	double Seconds() const { return elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1e6; }
	double Millisecs() const { return 1000*elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1000; }
	double Microsecs() const { return 1e6*elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec; }
private:
	struct timeval start_time_;
	struct timeval elapsed_time_;
};

