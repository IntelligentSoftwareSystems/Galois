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

inline void agg_func(const FV &in, FV &out) {
	for (size_t i = 0; i < out.size(); ++i) {
		out[i] += in[i];
	}
}

// user-defined aggregate function
void aggregate(Graph &g, const VertexID src, const FV2D &features, FV &sum) {
	unsigned degree = 0;
	for (const auto e : g.edges(src)) {
		const auto dst = g.getEdgeDst(e);
		agg_func(features[dst], sum);
		degree ++;
	}
	if (degree == 0) return;
	for (size_t i = 0; i < sum.size(); ++i) sum[i] /= degree; // average
}

// user-defined combine function
// N: number of vertices, D: feature vector dimentions, 
// E: number of distinct labels, i.e. number of vertex classes
// layer 1: features N x D, weights D x 16, out N x 16 (hidden1=16)
// layer 2: features N x 16, weights 16 x E, out N x E
void combine(FV2D &W, FV2D &Q, const FV &a, const FV &b, FV &out) {
	size_t m = out.size();
	FV c(m, 0);
	FV d(m, 0);
	mvmul(Q, a, c);
	mvmul(W, b, d);
	vadd(c, d, out);
}

// forward propogation
void forward(Graph &g, const FV2D &h_in, FV2D &h_hidden1, FV2D &h_out, FV3D W, FV3D Q, LabelList labels, MaskList masks, AccT &loss, AccT &accuracy) {
	//std::cout << "\n[debug] forward\n";
	size_t dim = h_in[0].size();
	//std::cout << "[debug] layer 0\n";
	galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
		FV h_neighbors(dim, 0); // used to gather neighbors' embeddings
		aggregate(g, src, h_in, h_neighbors);
		combine(W[0], Q[0], h_in[src], h_neighbors, h_hidden1[src]);
		relu(h_hidden1[src]);
	}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Layer0"));

	//std::cout << "[debug] layer 1\n";
	galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
		FV h_neighbors(hidden1, 0); // used to gather neighbors' embeddings
		aggregate(g, src, h_hidden1, h_neighbors);
		combine(W[1], Q[1], h_hidden1[src], h_neighbors, h_out[src]);
		relu(h_out[src]);
	}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Layer1"));

	auto n = g.size();
	//for (size_t i = 0; i < 3; i ++)
	//	for (size_t j = 0; j < h_out[0].size(); j++)
	//		std::cout << "out[" << i << "][" << j << "]=" << h_out[i][j] << "\n";
	//std::cout << "[debug] calculating loss and accuracy\n";
	loss = masked_softmax_cross_entropy(h_out, labels, masks);
	//std::cout << "loss: " << loss << "\n";
	// comparing outputs (N x E) with the ground truth (labels)
	LabelList predictions(n);
	auto num_classes = W[1][0].size();
	for (size_t i = 0; i < n; i ++) predictions[i] = argmax(num_classes, h_out[i]);
	//for (size_t i = 0; i < 10; i ++) std::cout << "predictions[" << i << "]=" << predictions[i] << "\n";
	accuracy = masked_accuracy(predictions, labels, masks);
	//std::cout << "accuracy: " << accuracy << "\n";
}

// evaluate, i.e. inference or predict
void evaluate(Graph &g, const FV2D &h_in, FV3D W, FV3D Q, LabelList labels, MaskList masks, AccT &loss, AccT &acc) {
	auto num_classes = W[1][0].size();
	auto n = g.size();
	FV2D h_hidden1(n); // hidden1 level embedding
	FV2D h_out(n); // out level embedding
	for (size_t i = 0; i < n; ++i) h_hidden1[i].resize(hidden1);
	for (size_t i = 0; i < n; ++i) h_out[i].resize(num_classes);
	//std::cout << "loss: " << loss << "\n";
	forward(g, h_in, h_hidden1, h_out, W, Q, labels, masks, loss, acc);
}

// back propogation
void backward(Graph &g, FV2D h_in, FV2D h_hidden1, FV2D h_out, FV3D &W, FV3D &Q) {
	auto n = g.size();
	FV2D in_diff(n);
	FV2D hidden1_diff(n);
	FV2D out_diff(n);
	galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
		d_relu(in_diff[src], h_out[src], out_diff[src]);
		//d_add();
		//d_mvmul();
		//d_mvmul();
	}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("layer1-back"));

	galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
		//d_relu(in_diff[src], h_out[src], out_diff[src]);
		//d_add();
		//d_mvmul();
		//d_mvmul();
	}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("layer0-back"));
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	read_graph(dataset, graph); 

	//std::cout << "\nReading features\n";
	size_t n = graph.size();
	FV2D input_features(n); // N x D
	auto feature_dim = read_features(dataset, input_features);
	//std::cout << "Number of feature dimensions: " << feature_dim << "\n";

	//std::cout << "\nReading labels\n";
	std::vector<LabelT> labels(n, 0); // labels for classification
	auto num_classes = read_labels(dataset, labels);
	//std::cout << "Number of vertex classes: " << num_classes << "\n";

	//std::cout << "\nReading masks\n";
	MaskList train_mask(n, 0), val_mask(n, 0), test_mask(n, 0);
	set_masks(n, train_mask, val_mask, test_mask);
	LabelList y_train(n), y_val(n), y_test(n);
	for (size_t i = 0; i < n; i ++) y_train[i] = (train_mask[i] == 1 ? labels[i] : -1);
	for (size_t i = 0; i < n; i ++) y_val[i] = (val_mask[i] == 1 ? labels[i] : -1);
	for (size_t i = 0; i < n; i ++) y_test[i] = (test_mask[i] == 1 ? labels[i] : -1);

	//std::cout << "\nInitializing parameters\n";
	FV3D W(NUM_LAYERS); // parameters to learn
	FV3D Q(NUM_LAYERS); // parameters to learn
	init_matrix(feature_dim, hidden1, W[0]);
	init_matrix(hidden1, num_classes, W[1]);
	init_matrix(feature_dim, hidden1, Q[0]);
	init_matrix(hidden1, num_classes, Q[1]);

	//std::cout << "\nAllocating intermediate data\n";
	FV2D hidden1_features(n); // hidden1 level embedding: N x 16
	FV2D output_features(n); // output embedding: N x E
	for (size_t i = 0; i < n; ++i) hidden1_features[i].resize(hidden1);
	for (size_t i = 0; i < n; ++i) output_features[i].resize(num_classes);

	ResourceManager rm;
	//std::cout << "\nStarting training\n";
	galois::StatTimer Ttrain("Train");
	Ttrain.start();
	Timer t_epoch;
	// run epoches
	for (size_t i = 0; i < epochs; i++) {
		std::cout << "Epoch " << i << ": ";
		t_epoch.Start();
		// Construct feed dictionary

		// Training step
		AccT train_loss = 0.0, train_acc = 0.0;
		forward(graph, input_features, hidden1_features, output_features, W, Q, y_train, train_mask, train_loss, train_acc);
		//backward(graph, input_features, hidden1_features, output_features, W, Q); // back propogation
		std::cout << " train_loss = " << train_loss << " train_acc = " << train_acc;

		// Validation
		AccT val_cost = 0.0, val_acc = 0.0;
		//Timer t_eval;
		//t_eval.Start();
		evaluate(graph, input_features, W, Q, y_val, val_mask, val_cost, val_acc);
		//t_eval.Stop();
		//t_eval.Millisecs();
		std::cout << " val_cost = " << val_cost << " val_acc = " << val_acc;

		t_epoch.Stop();
		std::cout << " time = " << t_epoch.Millisecs() << "ms. \n";
	}
	Ttrain.stop();

	AccT test_cost = 0.0, test_acc = 0.0;
	galois::StatTimer Ttest("Test");
	Ttest.start();
	evaluate(graph, input_features, W, Q, y_test, test_mask, test_cost, test_acc);
	std::cout << "\nTesting: test_loss = " << test_cost << " test_acc = " << test_acc << "\n";
	Ttest.stop();

	std::cout << "\n" << rm.get_peak_memory() << "\n\n";
	return 0;
}

