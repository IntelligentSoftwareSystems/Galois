#pragma once
#include "types.h"
#include "utils.h"
#include "cutils.h"
//#include "random.h"

class DeepGalois {
public:
	~DeepGalois();
	enum Brew { CPU, GPU };
	static DeepGalois& Get() {
	}
	inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
	inline static curandGenerator_t curand_generator() { return Get().curand_generator_; }
	inline static Brew mode() { return Get().mode_; }
	inline static void set_mode(Brew mode) { Get().mode_ = mode; }
	inline static int solver_count() { return Get().solver_count_; }
	inline static void set_solver_count(int val) { Get().solver_count_ = val; }
	inline static int solver_rank() { return Get().solver_rank_; }
	inline static void set_solver_rank(int val) { Get().solver_rank_ = val; }
	inline static bool multiprocess() { return Get().multiprocess_; }
	inline static void set_multiprocess(bool val) { Get().multiprocess_ = val; }
	inline static bool root_solver() { return Get().solver_rank_ == 0; }
	static void SetDevice(const int device_id) {
		int current_device;
		CUDA_CHECK(cudaGetDevice(&current_device));
		if (current_device == device_id) return;
		CUDA_CHECK(cudaSetDevice(device_id));
		if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
		if (Get().curand_generator_) CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
		CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
		//CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
		//CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_, cluster_seedgen()));
	}
	static void DeviceQuery() {}
	static bool CheckDevice(const int device_id) { return true; }
	static int FindDevice(const int start_id = 0) { return 0; }

protected:
	cublasHandle_t cublas_handle_; // used to call cuBLAS
	curandGenerator_t curand_generator_; // used to generate random numbers on GPU
	//shared_ptr<RNG> random_generator_;
	Brew mode_;
	// Parallel training
	int solver_count_;
	int solver_rank_;
	bool multiprocess_;

private:
	// The private constructor to avoid duplicate instantiation.
	DeepGalois() : cublas_handle_(NULL), curand_generator_(NULL), 
			//random_generator_(NULL), mode_(DeepGalois::CPU),
			mode_(DeepGalois::CPU),
			solver_count_(1), solver_rank_(0), multiprocess_(false) {
		// Try to create a cublas handler, and report an error if failed (but we will
		// keep the program running as one might just want to run CPU code).
		if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Cannot create Cublas handle. Cublas won't be available.";
		}
		// Try to create a curand handler.
		//if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS ||
		//	curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()) != CURAND_STATUS_SUCCESS)
		//	std::cout << "Cannot create Curand generator. Curand won't be available.";
	}
};

