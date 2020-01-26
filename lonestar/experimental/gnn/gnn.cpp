// Graph Neural Networks
// Xuhao Chen <cxh@utexas.edu>
#include "gnn.h"
#include "types.h"
#include "utils.h"
#include "math_functions.hpp"
#define NUM_LAYERS 2
#define CHUNK_SIZE 256

const char* name = "Graph Convolutional Networks";
const char* desc = "Graph convolutional neural networks on an undirected graph";
const char* url  = 0;

inline void agg_func(size_t dim, FV in, FV &out) {
	for (size_t i = 0; i < dim; ++i) {
		out[i] += in[i];
	}
}

// user-defined aggregate function
void aggregate(size_t dim, Graph &g, VertexID src, FV2D features, FV &sum) {
	unsigned degree = 0;
	for (auto e : g.edges(src)) {
		auto dst = g.getEdgeDst(e);
		agg_func(dim, features[dst], sum);
		degree ++;
	}
	for (size_t i = 0; i < dim; ++i) sum[i] /= degree; // average
}

// user-defined combine function
void combine(size_t dim, FV2D weights, FV2D biases, FV a, FV b, FV &out) {
	FV c(dim, 0);
	FV d(dim, 0);
	mvmul(dim, biases, a, c);
	mvmul(dim, weights, b, d);
	vadd(dim, c, d, out);
}

// forward propogation, i.e. inference
void forward(Graph &g, FV2D &features, FV3D &outs, FV3D weight_matrices, FV3D bias_matrices, LabelList labels, MaskList masks, double &loss, double &accuracy) {
	auto n = g.size();
	size_t dim = features[0].size();
	FV2D h_curr = features; // current level embedding (top)
	for (unsigned l = 0; l < NUM_LAYERS; l ++) {
		outs[l].resize(n);
		for (size_t i = 0; i < n; ++i) outs[l][i].resize(dim);
		galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
			FV h_neighbors(dim, 0); // used to gather neighbors' embeddings
			aggregate(dim, g, src, h_curr, h_neighbors);
			combine(dim, weight_matrices[l], bias_matrices[l], h_curr[src], h_neighbors, outs[l][src]);
			relu(outs[l][src]);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Encoder"));
		h_curr = outs[l];
	}
	features = outs[NUM_LAYERS-1];
	LabelList predictions;
	loss += masked_softmax_cross_entropy(predictions, labels, masks);
	accuracy = masked_accuracy(predictions, labels, masks);
}

// back propogation
void backward(Graph &g, FV2D features, FV3D outs, FV3D &weight_matrices, FV3D &bias_matrices) {
	auto n = g.size();
	FV2D in_diff = features; // current level embedding (bottom)
	FV2D out_diff; // next level embedding (top)
	out_diff.resize(n);
	for (unsigned l = NUM_LAYERS-1; l >= 0; --l) {
		FV2D weight_diff;
		out_diff.resize(n);
		galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& i) {
			//out_diff[i] = in_diff[i] * (outs[l][i] > 0);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Encoder"));
	}
}

void evaluate(Graph &g, FV2D &features, FV3D weight_matrices, FV3D bias_matrices, LabelList labels, MaskList masks, double &loss, double &accuracy) {
	auto n = g.size();
	size_t dim = features[0].size();
	FV2D h_curr = features; // current level embedding (top)
	FV2D h_next; // next level embedding (bottom)
	h_next.resize(n);
	for (size_t i = 0; i < n; ++i) h_next[i].resize(dim);
	for (unsigned l = 0; l < NUM_LAYERS; l ++) {
		galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
			FV h_neighbors(dim, 0); // used to gather neighbors' embeddings
			aggregate(dim, g, src, h_curr, h_neighbors);
			combine(dim, weight_matrices[l], bias_matrices[l], h_curr[src], h_neighbors, h_next[src]);
			relu(h_next[src]);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Encoder"));
		h_curr = h_next;
	}
	features = h_next;
	LabelList predictions;
	loss += masked_softmax_cross_entropy(predictions, labels, masks);
	accuracy = masked_accuracy(predictions, labels, masks);
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);

	Graph graph;

	FV2D input_features;
	//FV2D output_features; // vertex embeddings to get from inference
	LabelList y_train, y_val, y_test; // labels for classification
	load_data(dataset, graph, input_features); 
	size_t n = graph.size();
	std::vector<unsigned> labels(n, 0);
	load_labels(n, dataset, labels, y_train, y_val, y_test);
	MaskList train_mask(n, 0), val_mask(n, 0), test_mask(n, 0);
	set_masks(n, train_mask, val_mask, test_mask);
	return 0;
	size_t feature_dim = input_features[0].size();
	FV3D weight_matrices; // parameters to learn
	FV3D bias_matrices; // parameters to learn

	weight_matrices.resize(NUM_LAYERS);
	bias_matrices.resize(NUM_LAYERS);
	for (int i = 0; i < NUM_LAYERS; i ++) {
		init_matrix(feature_dim, feature_dim, weight_matrices[i]);
		init_matrix(feature_dim, feature_dim, bias_matrices[i]);
	}

	ResourceManager rm;
	galois::StatTimer Ttrain("Train");
	Ttrain.start();
	Timer t_epoch;
	// run epoches
	for (size_t i = 0; i < epochs; i++) {
		std::cout << "Epoch: " << i;
		t_epoch.Start();
		// Construct feed dictionary

		// Training step
		double train_loss, train_acc;
		FV3D output_features;
		forward(graph, input_features, output_features, weight_matrices, bias_matrices, y_train, train_mask, train_loss, train_acc);
		backward(graph, input_features, output_features, weight_matrices, bias_matrices); // back propogation
		std::cout << " train_loss = " << train_loss << " train_acc = " << train_acc;

		// Validation
		//double cost, acc;
		//Timer t_eval;
		//t_eval.Start();
		//evaluate(graph, input_features, weight_matrices, bias_matrices, y_val, val_mask, cost, acc);
		//std::cout << " val_loss = " << cost << " val_acc = " << acc 
		//t_eval.Stop();
		//t_eval.Millisecs();

		t_epoch.Stop();
		std::cout << " time = " << t_epoch.Millisecs() << "ms. \n";
	}
	Ttrain.stop();

	double test_cost, test_acc;
	galois::StatTimer Ttest("Test");
	Ttest.start();
	evaluate(graph, input_features, weight_matrices, bias_matrices, y_test, test_mask, test_cost, test_acc);
	Ttest.stop();
	std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
	return 0;
}

