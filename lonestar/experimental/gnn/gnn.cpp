// Graph Neural Networks
// Xuhao Chen <cxh@utexas.edu>
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/Profile.h"
#include <boost/iterator/transform_iterator.hpp>

namespace cll = llvm::cl;
//static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: unsymmetrized graph>"), cll::Required);
static cll::opt<unsigned> L("l", cll::desc("depth, i.e. number of layers (default value 3)"), cll::init(3));
static cll::opt<unsigned> K("k", cll::desc("number of epoch, i.e. iterations (default value 5)"), cll::init(5));
#ifdef EDGE_LABEL
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#else
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#endif
typedef Graph::GraphNode GNode;
typedef float FeatureT; // feature type
typedef std::vector<FeatureT> FV; // feature vector
typedef std::vector<FV> FV2D; // feature vectors
typedef std::vector<FV2D> FV3D; // matrices 
#define Dim 20 // vector dimension
#define CHUNK_SIZE 256

const char* name = "Graph Neural Networks";
const char* desc = "Graph neural networks on an undirected graph";
const char* url  = 0;

#include "util.h"
#include <random>
void init_features(size_t n, FV2D x) {
	x.resize(n);
	std::default_random_engine rng;
	std::uniform_real_distribution<float> dist(0, 0.1);
	for (size_t i = 0; i < n; ++i) {
		x[i].resize(Dim);
		for (size_t j = 0; j < Dim; ++j) {
			x[i][j] = dist(rng);
		}
	}
}

void init_matrices(unsigned l, FV3D x) {
	x.resize(l);
	std::default_random_engine rng;
	std::uniform_real_distribution<float> dist(0, 0.1);
	for (size_t i = 0; i < l; ++i) {
		x[i].resize(Dim);
		for (size_t j = 0; j < Dim; ++j) {
			x[i][j].resize(Dim);
			for (size_t k = 0; k < Dim; ++k) {
				x[i][j][k] = dist(rng);
			}
		}
	}
}

#include <cmath>
inline FeatureT sigmoid_func(FeatureT x) {
	return 0.5 * tanh(0.5 * x) + 0.5;
}

// vector add
void vadd(FV in_a, FV in_b, FV &out) {
	for (size_t i = 0; i < Dim; ++i) {
		out[i] = in_a[i] + in_b[i];
	}
}

// matrix-vector multiply
void mvmul(FV2D weights, FV input_vector, FV &output_vector) {
	for (size_t i = 0; i < Dim; ++i) { 
		for (size_t j = 0; j < Dim; ++j) { 
			output_vector[i] += weights[i][j] * input_vector[j];
		} 
	} 
}

// matrix multiply
void matmul(FV2D weights, FV2D input_vectors, FV2D &output_vectors) {
	for (size_t i = 0; i < Dim; ++i) { 
		for (size_t j = 0; j < Dim; ++j) { 
			for (size_t k = 0; k < Dim; ++k) { 
				output_vectors[i][j] += weights[i][k] * input_vectors[k][j];
			} 
		} 
	} 
}

// ReLU
const float negative_slope = 0;
void relu(FV &fv) {
	size_t count = fv.size();
	for (size_t i = 0; i < count; ++i) {
		fv[i] = std::max(fv[i], (FeatureT)0) + negative_slope * std::min(fv[i], (FeatureT)0);
	}
}

void sigmoid(FV fv) {
	size_t count = fv.size();
	for (size_t i = 0; i < count; ++i) {
		fv[i] = sigmoid_func(fv[i]);
	}
}

inline void agg_func(FV a, FV &b) {
	for (size_t i = 0; i < Dim; ++i) {
		b[i] += a[i];
	}
}

// user-defined aggregation function
void aggregate(Graph &g, VertexId src, FV2D embeddings, FV &sum) {
	unsigned degree = 0;
	for (auto e : g.edges(src)) {
		auto dst = g.getEdgeDst(e);
		agg_func(embeddings[dst], sum);
		degree ++;
	}
	for (size_t i = 0; i < Dim; ++i) sum[i] /= degree; // average
}

void combine(FV2D weights, FV2D biases, FV a, FV b, FV &out) {
	FV c(Dim, 0);
	FV d(Dim, 0);
	mvmul(biases, a, c);
	mvmul(weights, b, d);
	vadd(c, d, out);
}

void normalize(FV2D features) {
}

// forward propogation, i.e. inference
void forward(Graph &g, FV2D inputs, FV3D weight_matrices, FV3D bias_matrices, FV2D &outputs) {
	auto n = g.size();
	FV2D h_curr = inputs; // current level embedding
	FV2D h_next; // next level embedding
	h_next.resize(n);
	for (size_t i = 0; i < n; ++i) h_next[i].resize(Dim);
	for (unsigned l = 1; l < L; l ++) {
		galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
			FV h_neighbors(Dim, 0); // used to gather neighbors' embeddings
			aggregate(g, src, h_curr, h_neighbors);
			combine(weight_matrices[l], bias_matrices[l], h_curr[src], h_neighbors, h_next[src]);
			relu(h_next[src]);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Encoder"));
		//normalize(h_next);
	}
	outputs = h_next;
}

// back propogation
void backward(Graph &g, FV2D inputs, FV2D outputs, FV3D &weight_matrices, FV3D &bias_matrices) {
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	galois::StatTimer Tinitial("GraphReadingTime");
	printf("Start readGraph\n");
	Tinitial.start();
    galois::graphs::readGraph(graph, filename);
	Tinitial.stop();
	printf("Done readGraph\n");
	std::cout << "num_vertices " << graph.size() << " num_edges " << graph.sizeEdges() << "\n";
	
	auto n = graph.size();
	FV2D input_features;
	FV2D output_features; // vertex embeddings to get from inference
	FV3D weight_matrices; // parameters to learn
	FV3D bias_matrices; // parameters to learn
	init_features(n, input_features);
	init_matrices(L, weight_matrices);
	init_matrices(L, bias_matrices);

	ResourceManager rm;
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	// run K epoches
	for (size_t i = 0; i < K; i++) {
		forward(graph, input_features, weight_matrices, bias_matrices, output_features); // forward-propogation, i.e. inference
		backward(graph, input_features, output_features, weight_matrices, bias_matrices); // back propogation
	}
	Tcomp.stop();
	//print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

