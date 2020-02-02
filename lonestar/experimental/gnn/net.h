#ifndef _MODEL_H_
#define _MODEL_H_

#include <random>
#include "types.h"
#include "utils.h"
#include "lgraph.h"
#include "layers.h"
#include "optimizer.h"
#include "math_functions.hpp"

#define NUM_CONV_LAYERS 2
std::string path = "/h2/xchen/datasets/Learning/"; // path to the input dataset

// N: number of vertices, D: feature vector dimentions, 
// E: number of distinct labels, i.e. number of vertex classes
// layer 1: features N x D, weights D x 16, out N x 16 (hidden1=16)
// layer 2: features N x 16, weights 16 x E, out N x E
class Net {
public:
	Net() {}

	// user-defined aggregate function
	virtual void aggregate(size_t dim, const FV2D &features, FV2D &sum) {}
	
	// user-defined combine function
	virtual void combine(const vec_t ma, const vec_t mb, const FV &a, const FV &b, FV &out) {}
	
	void init() {
		read_graph(dataset, g); 
		n = g.size(); // N
		labels.resize(n, 0); // label for each vertex: N x 1
		num_classes = read_labels(dataset, labels);

		train_mask.resize(n, 0);
		val_mask.resize(n, 0);
		set_masks(n, train_mask, val_mask);
		y_train.resize(n, 0);
		y_val.resize(n, 0);
		for (size_t i = 0; i < n; i ++) y_train[i] = (train_mask[i] == 1 ? labels[i] : -1);
		for (size_t i = 0; i < n; i ++) y_val[i] = (val_mask[i] == 1 ? labels[i] : -1);

		num_layers = NUM_CONV_LAYERS + 1;
		features.resize(num_layers + 1);
		features[0].resize(n); // input embedding: N x D
		input_feature_dim = read_features(dataset, features[0]);

		feature_dims.resize(num_layers + 1);
		feature_dims[0] = input_feature_dim; // input feature dimension
		feature_dims[1] = hidden1; // hidden1 level embedding: N x 16
		feature_dims[2] = num_classes; // output embedding: N x E
		feature_dims[3] = num_classes; // normalized output embedding: N x E
		/*
		for (size_t i = 1; i < num_layers + 1; i ++) {
			features[i].resize(n); 
			for (size_t j = 0; j < n; ++j)
				features[i][j].resize(feature_dims[i]);
		}
		*/
		diffs.resize(n);
		layers.resize(num_layers);
		std::vector<size_t> in_dims(2), out_dims(2);
		in_dims[0] = out_dims[0] = n;
		for (size_t i = 0; i < NUM_CONV_LAYERS; ++i) {
			in_dims[1] = feature_dims[i];
			out_dims[1] = feature_dims[i+1];
			layers[i] = new graph_conv_layer(i, &g, in_dims, out_dims);
		}
		layers[0]->set_act(true);
		layers[0]->set_in_data(features[0]);
		in_dims[1] = feature_dims[2];
		out_dims[1] = feature_dims[3];
		connect(layers[0], layers[1]);
		layers[2] = new softmax_loss_layer(2, in_dims, out_dims, &diffs, &labels);
		connect(layers[1], layers[2]);

		for (size_t i = 0; i < num_layers; i ++)
			layers[i]->print_layer_info();
	}
	size_t get_nnodes() { return n; }
	size_t get_nedges() { return g.sizeEdges(); }
	size_t get_ft_dim() { return input_feature_dim; }
	size_t get_nclasses() { return num_classes; }
	size_t get_label(size_t i) { return labels[i]; }

	// forward propagation
	void fprop(LabelList labels, MaskList masks, acc_t &loss, acc_t &accuracy) {
		// layer0: from N x D to N x 16
		// layer1: from N x 16 to N x E
		// layer2: from N x E to N x E (normalize only)
		for (size_t i = 0; i < num_layers; i ++)
			layers[i]->forward();
		loss = masked_avg_loss(diffs, masks);

		// comparing outputs (N x E) with the ground truth (labels)
		LabelList predictions(n);
		for (size_t i = 0; i < n; i ++)
			//predictions[i] = argmax(num_classes, features[NUM_CONV_LAYERS][i]);
			predictions[i] = argmax(num_classes, layers[NUM_CONV_LAYERS-1]->next()->get_data()[i]);
		accuracy = masked_accuracy(predictions, labels, masks);
	}

	// back propogation
	void bprop() {
		for (size_t i = num_layers; i != 0; i --)
			layers[i-1]->backward();
	}
	void update_weights(optimizer *opt) {
		for (size_t i = 0; i < num_layers; i ++)
			if (layers[i]->trainable()) layers[i]->update_weight(opt);
	}

	// evaluate, i.e. inference or predict
	double evaluate(LabelList labels, MaskList masks, acc_t &loss, acc_t &acc) {
		Timer t_eval;
		t_eval.Start();
		fprop(labels, masks, loss, acc);
		t_eval.Stop();
		return t_eval.Millisecs();
	}

	void train(optimizer *opt) {
		std::cout << "\nStart training...\n";
		Timer t_epoch;
		// run epoches
		for (size_t i = 0; i < epochs; i++) {
			std::cout << "Epoch " << i << ":";
			t_epoch.Start();
			// Construct feed dictionary

			// Training step
			acc_t train_loss = 0.0, train_acc = 0.0;
			fprop(y_train, train_mask, train_loss, train_acc);
			bprop(); // back propogation
			update_weights(opt);
			std::cout << " train_loss = " << train_loss << " train_acc = " << train_acc;

			// Validation
			acc_t val_cost = 0.0, val_acc = 0.0;
			double eval_time = evaluate(y_val, val_mask, val_cost, val_acc);
			std::cout << " val_cost = " << val_cost << " val_acc = " << val_acc;

			t_epoch.Stop();
			std::cout << " time = " << t_epoch.Millisecs() << "ms\n";
		}
	}

protected:
	size_t n; // N
	size_t input_feature_dim; // D
	size_t num_classes; // E
	size_t num_layers; // for now hard-coded: NUM_CONV_LAYERS + 1
	std::vector<size_t> feature_dims;

	Graph g; // the input graph
	FV3D features; // features: num_layers x N x (D; 16; E)
	FV3D gradients; // gradient: 16 x E; N x E; D x 16
	std::vector<acc_t> diffs; // error for each vertex: N

	std::vector<label_t> labels; // labels for classification
	LabelList y_train, y_val; // labels for traning and validation
	MaskList train_mask, val_mask; // masks for traning and validation

	std::vector<layer *> layers; // all the layers in the neural network

	inline void init_features(size_t dim, FV &x) {
		std::default_random_engine rng;
		std::uniform_real_distribution<feature_t> dist(0, 0.1);
		for (size_t i = 0; i < dim; ++i)
			x[i] = dist(rng);
	}

	// labels contains the ground truth (e.g. vertex classes) for each example (num_examples x 1).
	// Note that labels is not one-hot encoded vector,
	// and it can be computed as y.argmax(axis=1) from one-hot encoded vector (y) of labels if required.
	size_t read_labels(std::string dataset_str, LabelList &labels) {
		std::string filename = path + dataset_str + "-labels.txt";
		std::ifstream in;
		std::string line;
		in.open(filename, std::ios::in);
		size_t m, n;
		in >> m >> n >> std::ws;
		assert(m == labels.size()); // number of vertices
		std::cout << "label conuts: " << n << std::endl; // number of vertex classes
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

		//for (size_t i = 0; i < 10; ++i)
		//	std::cout << "labels[" << i << "]: " << labels[i] << std::endl;
		return n;
	}

	size_t read_features(std::string dataset_str, FV2D &features) {
		std::string filename = path + dataset_str + ".ft";
		std::ifstream in;
		std::string line;
		in.open(filename, std::ios::in);
		size_t m, n;
		in >> m >> n >> std::ws;
		assert(m == features.size()); // m = number of vertices
		std::cout << "feature dimention: " << n << std::endl;
		for (size_t i = 0; i < m; ++i) {
			features[i].resize(n);
			for (size_t j = 0; j < n; ++j)
				features[i][j] = 0;
		}
		while (std::getline(in, line)) {
			std::istringstream edge_stream(line);
			unsigned u, v;
			float_t w;
			edge_stream >> u;
			edge_stream >> v;
			edge_stream >> w;
			features[u][v] = w;
		}
		/*
		for (size_t i = 0; i < 10; ++i) {
			for (size_t j = 0; j < n; ++j) {
				if (features[i][j] > 0)
					std::cout << "features[" << i << "][" << j << "]: " << features[i][j] << std::endl;
			}
		}
		//*/
		return n;
	}

	unsigned load_graph(Graph &graph, std::string filename, std::string filetype = "el") {
		LGraph lgraph;
		unsigned max_degree = 0;
		if (filetype == "el") {
			printf("Reading .el file: %s\n", filename.c_str());
			lgraph.read_edgelist(filename.c_str(), true); //symmetrize
			genGraph(lgraph, graph);
		} else if (filetype == "gr") {
			printf("Reading .gr file: %s\n", filename.c_str());
			galois::graphs::readGraph(graph, filename);
			galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const auto& vid) {
				graph.getData(vid) = 1;
				//for (auto e : graph.edges(n)) graph.getEdgeData(e) = 1;
			}, galois::chunk_size<256>(), galois::steal(), galois::loopname("assignVertexLabels"));
			std::vector<unsigned> degrees(graph.size());
			galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const auto& vid) {
				degrees[vid] = std::distance(graph.edge_begin(vid), graph.edge_end(vid));
			}, galois::loopname("computeMaxDegree"));
			max_degree = *(std::max_element(degrees.begin(), degrees.end()));
		} else { printf("Unkown file format\n"); exit(1); }
		if (filetype != "gr") {
			max_degree = lgraph.get_max_degree();
			lgraph.clean();
		}
		printf("max degree = %u\n", max_degree);
		return max_degree;
	}

	void genGraph(LGraph &lg, Graph &g) {
		g.allocateFrom(lg.num_vertices(), lg.num_edges());
		g.constructNodes();
		for (size_t i = 0; i < lg.num_vertices(); i++) {
			g.getData(i) = 1;
			auto row_begin = lg.get_offset(i);
			auto row_end = lg.get_offset(i+1);
			g.fixEndEdge(i, row_end);
			for (auto offset = row_begin; offset < row_end; offset ++)
				g.constructEdge(offset, lg.get_dest(offset), 0); // do not consider edge labels currently
		}
	}

	void read_graph(std::string dataset_str, Graph &g) {
		//printf("Start readGraph\n");
		galois::StatTimer Tread("GraphReadingTime");
		Tread.start();
		//std::string filename = dataset_str + ".gr";
		std::string filename = path + dataset_str + ".el";
		load_graph(g, filename);
		Tread.stop();
		//printf("Done readGraph\n");
		std::cout << "num_vertices " << g.size() << " num_edges " << g.sizeEdges() << "\n";
	}

	void set_masks(size_t n, MaskList &train_mask, MaskList &val_mask) {
		for (size_t i = 0; i < n; i++) {
			if (i < 120) train_mask[i] = 1; // [0, 120) train size = 120
			else if (i < 620) val_mask[i] = 1; // [120, 620) validation size = 500
			else ; // unlabeled vertices
		}
	}

};

#endif
