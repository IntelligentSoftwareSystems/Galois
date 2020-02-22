#pragma once
#include <string>
#include <cassert>
#include "types.h"
#include "utils.h"
#include "lgraph.h"
#include "gtypes.h"
#include "cutils.h"
//#include "random.h"

class Context {
public:
	Context();
	~Context();
	enum Brew { CPU, GPU };
	//static Context& Get();
	cublasHandle_t cublas_handle() { return cublas_handle_; }
	curandGenerator_t curand_generator() { return curand_generator_; }
	Brew mode() { return mode_; }
	void set_mode(Brew mode) { mode_ = mode; }
	int solver_count() { return solver_count_; }
	void set_solver_count(int val) { solver_count_ = val; }
	int solver_rank() { return solver_rank_; }
	void set_solver_rank(int val) { solver_rank_ = val; }
	bool multiprocess() { return multiprocess_; }
	void set_multiprocess(bool val) { multiprocess_ = val; }
	bool root_solver() { return solver_rank_ == 0; }
	void SetDevice(const int device_id);
	void DeviceQuery() {}
	bool CheckDevice(const int device_id) { return true; }
	int FindDevice(const int start_id = 0) { return 0; }
	size_t read_graph(std::string dataset_str);
	size_t read_graph_cpu(std::string dataset_str, std::string filetype = "gr");
	size_t read_graph_gpu(std::string dataset_str);
	size_t read_labels(std::string dataset_str, size_t num);
	label_t get_label(size_t i) { return labels[i]; }
	label_t *get_labels_ptr(size_t i) { return &(labels[0]); }
	void degree_counting();
	void norm_factor_counting();
#ifdef CPU_ONLY
	Graph graph_cpu; // the input graph, |V| = N
#else
	CSRGraph graph_gpu; // the input graph, |V| = N
#endif
	std::vector<label_t> labels; // labels for classification: N x 1
	std::vector<float_t> norm_factor; // normalization constant based on graph structure
	std::vector<unsigned> degrees;

protected:
	Brew mode_;
	cublasHandle_t cublas_handle_; // used to call cuBLAS
	curandGenerator_t curand_generator_; // used to generate random numbers on GPU
	//shared_ptr<RNG> random_generator_;
	// Parallel training
	int solver_count_;
	int solver_rank_;
	bool multiprocess_;
	void genGraph(LGraph &lg, Graph &g);

private:
	// The private constructor to avoid duplicate instantiation.
	//Context();
};

