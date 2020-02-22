#include "context.h"
#include <cstdio>
#include <ctime>

// random seeding
int64_t cluster_seedgen(void) {
	int64_t s, seed, pid;
	FILE* f = fopen("/dev/urandom", "rb");
	if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
		fclose(f);
		return seed;
	}
	std::cout << "System entropy source not available, "
		"using fallback algorithm to generate seed instead.";
	if (f) fclose(f);
	pid = getpid();
	s = time(NULL);
	seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
	return seed;
}

void Context::SetDevice(const int device_id) {
	int current_device;
	CUDA_CHECK(cudaGetDevice(&current_device));
	if (current_device == device_id) return;
	CUDA_CHECK(cudaSetDevice(device_id));
	if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
	if (curand_generator_) CURAND_CHECK(curandDestroyGenerator(curand_generator_));
	CUBLAS_CHECK(cublasCreate(&cublas_handle_));
	CURAND_CHECK(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()));
}

Context::Context() : 
		mode_(Context::CPU),
		cublas_handle_(NULL), curand_generator_(NULL), 
		//random_generator_(NULL), mode_(Context::CPU),
		solver_count_(1), solver_rank_(0), multiprocess_(false) {
#ifndef CPU_ONLY
	mode_ = Context::GPU;
	// Try to create a cublas handler, and report an error if failed (but we will
	// keep the program running as one might just want to run CPU code).
	if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
		std::cout << "Cannot create Cublas handle. Cublas won't be available.";
	}
	// Try to create a curand handler.
	if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS ||
		curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()) != CURAND_STATUS_SUCCESS)
		std::cout << "Cannot create Curand generator. Curand won't be available.";
#endif
}

size_t Context::read_graph(std::string dataset_str) {
#ifdef CPU_ONLY
	size_t n = read_graph_cpu(dataset_str, "gr");
#else
	size_t n = read_graph_gpu(dataset_str);
#endif
	return n;
}

size_t Context::read_graph_cpu(std::string dataset_str, std::string filetype) {
	galois::StatTimer Tread("GraphReadingTime");
	Tread.start();
	LGraph lgraph;
	if (filetype == "el") {
		std::string filename = path + dataset_str + ".el";
		printf("Reading .el file: %s\n", filename.c_str());
		lgraph.read_edgelist(filename.c_str(), true); //symmetrize
		genGraph(lgraph, graph_cpu);
		lgraph.clean();
	} else if (filetype == "gr") {
		std::string filename = path + dataset_str + ".csgr";
		printf("Reading .gr file: %s\n", filename.c_str());
		galois::graphs::readGraph(graph_cpu, filename);
	} else { printf("Unkown file format\n"); exit(1); }
	Tread.stop();
	std::cout << "num_vertices " << graph_cpu.size() << " num_edges " << graph_cpu.sizeEdges() << "\n";
	return graph_cpu.size();
}

size_t Context::read_graph_gpu(std::string dataset_str) {
}

void Context::genGraph(LGraph &lg, Graph &g) {
	g.allocateFrom(lg.num_vertices(), lg.num_edges());
	g.constructNodes();
	for (size_t i = 0; i < lg.num_vertices(); i++) {
		g.getData(i) = 1;
		auto row_begin = lg.get_offset(i);
		auto row_end = lg.get_offset(i+1);
		g.fixEndEdge(i, row_end);
		for (auto offset = row_begin; offset < row_end; offset ++)
			g.constructEdge(offset, lg.get_dest(offset), 0);
	}
}

// user-defined pre-computing function, called during initialization
// for each vertex v, compute pow(|N(v)|, -0.5), where |N(v)| is the degree of v
void Context::norm_factor_counting() {
#ifdef CPU_ONLY
	size_t n = graph_cpu.size();
	norm_factor.resize(n);
	galois::do_all(galois::iterate((size_t)0, n), [&] (auto v) {
		float_t temp = std::sqrt(float_t(degrees[v]));
		if (temp == 0.0) norm_factor[v] = 0.0;
		else norm_factor[v] = 1.0 / temp;
	}, galois::loopname("NormCounting"));
#endif
}

void Context::degree_counting() {
#ifdef CPU_ONLY
	size_t n = graph_cpu.size();
	degrees.resize(n);
	galois::do_all(galois::iterate((size_t)0, n), [&] (auto v) {
		degrees[v] = std::distance(graph_cpu.edge_begin(v), graph_cpu.edge_end(v));
	}, galois::loopname("DegreeCounting"));
#endif
}

// labels contain the ground truth (e.g. vertex classes) for each example (num_examples x 1).
// Note that labels is not one-hot encoded vector and it can be computed
// as y.argmax(axis=1) from one-hot encoded vector (y) of labels if required.
size_t Context::read_labels(std::string dataset_str, size_t num) {
	std::cout << "Reading labels ... ";
	labels.resize(num, 0); // label for each vertex: N x 1
	Timer t_read;
	t_read.Start();
	std::string filename = path + dataset_str + "-labels.txt";
	std::ifstream in;
	std::string line;
	in.open(filename, std::ios::in);
	size_t m, n;
	in >> m >> n >> std::ws;
	assert(m == labels.size()); // number of vertices
	unsigned v = 0;
	while (std::getline(in, line)) {
		std::istringstream label_stream(line);
		unsigned x;
		for (size_t idx = 0; idx < n; ++idx) {
			label_stream >> x;
			if (x != 0) {
				labels[v] = idx;
				break;
			}
		}
		v ++;
	}
	in.close();
	t_read.Stop();
	// number of vertex classes
	std::cout << "Done, unique label counts: " << n << ", time: " << t_read.Millisecs() << " ms\n";
	return n;
}

