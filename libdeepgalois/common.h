#pragma once
#include "cutils.h"

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
		CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_, cluster_seedgen()));
	}
	static void DeviceQuery();
	static bool CheckDevice(const int device_id);
	static int FindDevice(const int start_id = 0);

protected:
	cublasHandle_t cublas_handle_;
	curandGenerator_t curand_generator_;
	shared_ptr<RNG> random_generator_;
	Brew mode_;
	// Parallel training
	int solver_count_;
	int solver_rank_;
	bool multiprocess_;

private:
	// The private constructor to avoid duplicate instantiation.
	DeepGalois();
};

