#pragma once
#include <string>
#include <cassert>
#include "types.h"
#include "utils.h"
#include "lgraph.h"
#ifdef CPU_ONLY
#include "gtypes.h"
#else
#include "graph_gpu.h"
#endif
#include "cutils.h"

class Context {
public:
	Context();
	~Context();
	enum Brew { CPU, GPU };
	//static Context& Get();
#ifndef CPU_ONLY
	inline static cublasHandle_t cublas_handle() { return cublas_handle_; }
	inline static curandGenerator_t curand_generator() { return curand_generator_; }
	//static void create_blas_handle();
#endif
	Brew mode() { return mode_; }
	void set_mode(Brew mode) { mode_ = mode; }
	int solver_count() { return solver_count_; }
	void set_solver_count(int val) { solver_count_ = val; }
	int solver_rank() { return solver_rank_; }
	void set_solver_rank(int val) { solver_rank_ = val; }
	bool multiprocess() { return multiprocess_; }
	void set_multiprocess(bool val) { multiprocess_ = val; }
	bool root_solver() { return solver_rank_ == 0; }
	size_t read_graph(std::string dataset_str);
	size_t read_labels(std::string dataset_str);
	size_t read_features(std::string dataset_str);
	label_t get_label(size_t i) { return labels[i]; }
	label_t *get_labels_ptr(size_t i) { return &(labels[0]); }
	void degree_counting();
	void norm_factor_counting();
	std::vector<label_t> labels; // labels for classification: N x 1
	float_t *norm_factor; // normalization constant based on graph structure
	std::vector<unsigned> degrees;
	vec_t h_feats; // input features: N x D
	size_t n; // number of samples: N
	size_t num_classes; // number of classes: E
	size_t feat_len; // input feature length: D
#ifdef CPU_ONLY
	Graph graph_cpu; // the input graph, |V| = N
	void genGraph(LGraph &lg, Graph &g);
	size_t read_graph_cpu(std::string dataset_str, std::string filetype = "gr");
#else
	CSRGraph graph_gpu; // the input graph, |V| = N
	label_t *d_labels; // labels on device
	float_t *d_norm_factor; // norm_factor on device
	float_t *d_feats; // input features on device
	size_t read_graph_gpu(std::string dataset_str);
	void copy_data_to_device(); // copy labels and input features
	void SetDevice(const int device_id);
	void DeviceQuery() {}
	bool CheckDevice(const int device_id) { return true; }
	int FindDevice(const int start_id = 0) { return 0; }
#endif

protected:
#ifndef CPU_ONLY
	static cublasHandle_t cublas_handle_; // used to call cuBLAS
	static curandGenerator_t curand_generator_; // used to generate random numbers on GPU
#endif
	Brew mode_;
	//shared_ptr<RNG> random_generator_;
	// Parallel training
	int solver_count_;
	int solver_rank_;
	bool multiprocess_;

private:
	// The private constructor to avoid duplicate instantiation.
	//Context();
};

