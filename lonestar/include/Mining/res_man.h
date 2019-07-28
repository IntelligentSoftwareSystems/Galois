#ifndef RESOURCE_MANAGER_HPP_
#define RESOURCE_MANAGER_HPP_

#include <sys/time.h>
#include <sys/resource.h>
#include <iomanip>

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
#endif
