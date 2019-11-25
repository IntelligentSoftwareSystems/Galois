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
static cll::opt<unsigned> K("k", cll::desc("number of itertions (default value 3)"), cll::init(3));
#ifdef EDGE_LABEL
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#else
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#endif
typedef Graph::GraphNode GNode;
typedef float FeatureT;
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

#include <cmath>
inline FeatureT sigmoid_func(FeatureT x) {
	return 0.5 * tanh(0.5 * x) + 0.5;
}

void aggregate(FV a, FV b) {
}

void combine(FV a, FV b) {
}

void matmul(FV2D weights, FV input_vector, FV output_vector) {
	for (size_t i = 0; i < Dim; ++i) { 
		for (size_t j = 0; j < Dim; ++j) { 
			output_vector[i] += weights[i][j] * input_vector[j];
		} 
	} 
}

const float negative_slope = 0;
void relu(FV fv) {
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

void normalize(FV2D features) {
}

void encoder(Graph &g, FV2D inputs, FV3D weight_matrices, FV2D outputs) {
	auto n = g.size();
	FV2D h_curr = inputs;
	FV2D h_next;
	h_next.resize(n);
	for (size_t i = 0; i < n; ++i) h_next[i].resize(Dim);
	for (unsigned l = 1; l < K; l ++) {
		galois::do_all(galois::iterate(g.begin(), g.end()),
			[&](const auto& src) {
				//auto& src_label = g.getData(src);
				FV h_src = h_curr[src];
				FV h_neighbors(Dim, 0);
				for (auto e : g.edges(src)) {
					auto dst = g.getEdgeDst(e);
					FV h_dst = h_curr[dst];
					aggregate(h_dst, h_neighbors);
				}
				combine(h_src, h_neighbors);
				matmul(weight_matrices[l], h_src, h_next[src]);
				relu(h_next[src]);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("Encoder")
		);
		normalize(h_next);
	}
	outputs = h_next;
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
	FV2D output_features;
	FV3D weight_matrices;
	init_features(n, input_features);

	ResourceManager rm;
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	encoder(graph, input_features, weight_matrices, output_features);
	Tcomp.stop();
	//print_output();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

