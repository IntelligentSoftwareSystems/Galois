#ifndef _MODEL_H_
#define _MODEL_H_

#include <random>
#include "gnn.h"
#include "lgraph.h"
#include "layers.h"
#include "optimizer.h"

#define NUM_CONV_LAYERS 2

// N: number of vertices, D: feature vector dimentions, 
// E: number of distinct labels, i.e. number of vertex classes
// layer 1: features N x D, weights D x 16, out N x 16 (hidden1=16)
// layer 2: features N x 16, weights 16 x E, out N x E
class Net {
public:
	Net() {}

	// user-defined aggregate function
	virtual void aggregate(Graph *g, size_t dim, const tensor_t &in_feats, tensor_t &out_feats) {}
	
	// user-defined combine function
	virtual void combine(const vec_t ma, const vec_t mb, const vec_t &a, const vec_t &b, vec_t &out) {}
	
	void init() {
		assert(dropout_rate < 1.0);
		read_graph(dataset, g); 
		n = g.size(); // N
		labels.resize(n, 0); // label for each vertex: N x 1
		num_classes = read_labels(dataset, labels);

		std::cout << "Reading label masks ... ";
		train_mask.resize(n, 0);
		val_mask.resize(n, 0);
		if (dataset == "reddit") {
			train_begin = 0, train_count = 153431, train_end = train_begin + train_count;
			val_begin = 153431, val_count = 23831, val_end = val_begin + val_count;
			for (size_t i = train_begin; i < train_end; i++) train_mask[i] = 1;
			for (size_t i = val_begin; i < val_end; i++) val_mask[i] = 1;
		} else {
			train_count = read_masks(dataset, "train", train_begin, train_end, train_mask);
			val_count = read_masks(dataset, "val", val_begin, val_end, val_mask);
		}
		std::cout << "Done\n";

		num_layers = NUM_CONV_LAYERS + 1;
		feature_dims.resize(num_layers + 1);
		input_features.resize(n); // input embedding: N x D
		feature_dims[0] = read_features(dataset, input_features); // input feature dimension: D
		feature_dims[1] = hidden1; // hidden1 level embedding: 16
		feature_dims[2] = num_classes; // output embedding: E
		feature_dims[3] = num_classes; // normalized output embedding: E
		layers.resize(num_layers);
	}
	size_t get_in_dim(size_t layer_id) { return feature_dims[layer_id]; }
	size_t get_out_dim(size_t layer_id) { return feature_dims[layer_id+1]; }
	size_t get_nnodes() { return n; }
	size_t get_nedges() { return g.sizeEdges(); }
	size_t get_ft_dim() { return feature_dims[0]; }
	size_t get_nclasses() { return num_classes; }
	size_t get_label(size_t i) { return labels[i]; }
	void construct_layers() {
		std::cout << "\nConstructing layers...\n";
		append_conv_layer(0, true); // first conv layer
		append_conv_layer(1); // hidden1 layer
		append_out_layer(2); // output layer
		layers[0]->set_in_data(input_features); // feed input data
	}

	void set_netphase(net_phase phase) {
		for (size_t i = 0; i < num_layers; i ++)
			layers[i]->set_context(phase);
	}

	void print_layers_info() {
		for (size_t i = 0; i < num_layers; i ++)
			layers[i]->print_layer_info();
	}

	void append_conv_layer(size_t layer_id, bool act = false, bool norm = true, bool bias = false, bool dropout = true) {
		assert(layer_id < NUM_CONV_LAYERS);
		std::vector<size_t> in_dims(2), out_dims(2);
		in_dims[0] = out_dims[0] = n;
		in_dims[1] = get_in_dim(layer_id);
		out_dims[1] = get_out_dim(layer_id);
		layers[layer_id] = new graph_conv_layer(layer_id, &g, act, norm, bias, dropout, in_dims, out_dims);
		if(layer_id > 0) connect(layers[layer_id-1], layers[layer_id]);
	}

	void append_out_layer(size_t layer_id) {
		assert(layer_id > 0); // can not be the first layer
		std::vector<size_t> in_dims(2), out_dims(2);
		in_dims[0] = out_dims[0] = n;
		in_dims[1] = get_in_dim(layer_id);
		out_dims[1] = get_out_dim(layer_id);
		layers[layer_id] = new softmax_loss_layer(layer_id, in_dims, out_dims, &labels);
		connect(layers[layer_id-1], layers[layer_id]);
	}

	// forward propagation: [begin, end) is the range of samples used.
	acc_t fprop(size_t begin, size_t end, size_t count, MaskList &masks) {
		// set mask for the last layer
		layers[num_layers-1]->set_sample_mask(begin, end, count, masks);
		// layer0: from N x D to N x 16
		// layer1: from N x 16 to N x E
		// layer2: from N x E to N x E (normalize only)
		for (size_t i = 0; i < num_layers; i ++)
			layers[i]->forward();
		return layers[num_layers-1]->get_masked_loss();
	}

	// back propogation
	void bprop() {
		for (size_t i = num_layers; i != 0; i --)
			layers[i-1]->backward();
	}

	// update trainable weights after back-propagation
	void update_weights(optimizer *opt) {
		for (size_t i = 0; i < num_layers; i ++)
			if (layers[i]->trainable()) layers[i]->update_weight(opt);
	}

	// evaluate, i.e. inference or predict
	double evaluate(size_t begin, size_t end, size_t count, MaskList &masks, acc_t &loss, acc_t &acc) {
		Timer t_eval;
		t_eval.Start();
		loss = fprop(begin, end, count, masks);
		acc = masked_accuracy(begin, end, count, masks);
		t_eval.Stop();
		return t_eval.Millisecs();
	}

	// training
	void train(optimizer *opt) {
		std::cout << "\nStart training...\n";
		galois::StatTimer Tupdate("Train-WeightUpdate");
		galois::StatTimer Tfw("Train-Forward");
		galois::StatTimer Tbw("Train-Backward");
		galois::StatTimer Tval("Validation");
		Timer t_epoch;
		// run epoches
		for (size_t i = 0; i < epochs; i++) {
			std::cout << "Epoch " << std::setw(2) << i << std::fixed << std::setprecision(3) << ":";
			t_epoch.Start();

			// training steps
			set_netphase(net_phase::train);
			acc_t train_loss = 0.0, train_acc = 0.0;
			Tfw.start();
			train_loss = fprop(train_begin, train_end, train_count, train_mask); // forward
			train_acc = masked_accuracy(train_begin, train_end, train_count, train_mask); // predict
			Tfw.stop();
			Tbw.start();
			bprop(); // back propogation
			Tbw.stop();
			Tupdate.start();
			update_weights(opt); // update parameters
			Tupdate.stop();
			set_netphase(net_phase::test);
			std::cout << " train_loss = " << std::setw(5) << train_loss << " train_acc = " << std::setw(5) << train_acc;
			t_epoch.Stop();
			double epoch_time = t_epoch.Millisecs();

			if (do_validate) {
				// Validation
				acc_t val_loss = 0.0, val_acc = 0.0;
				Tval.start();
				double val_time = evaluate(val_begin, val_end, val_count, val_mask, val_loss, val_acc);
				Tval.stop();
				std::cout << " val_loss = " << std::setw(5) << val_loss << " val_acc = " << std::setw(5) << val_acc;
				std::cout << " time = " << epoch_time + val_time << " ms (train_time = " << epoch_time << " val_time = " << val_time << ")\n";
			} else {
				std::cout << " train_time = " << epoch_time << " ms\n";
			}
		}
	}

protected:
	size_t n; // number of samples: N
	size_t num_classes; // number of vertex classes: E
	size_t num_layers; // for now hard-coded: NUM_CONV_LAYERS + 1
	std::vector<size_t> feature_dims; // feature dimnesions for each layer

	Graph g; // the input graph, |V| = N
	tensor_t input_features; // input features: N x D
	std::vector<label_t> labels; // labels for classification: N x 1
	MaskList train_mask, val_mask; // masks for traning and validation
	size_t train_begin, train_end, train_count, val_begin, val_end, val_count;

	std::vector<layer *> layers; // all the layers in the neural network
	/*
	inline void init_features(size_t dim, vec_t &x) {
		std::default_random_engine rng;
		std::uniform_real_distribution<feature_t> dist(0, 0.1);
		for (size_t i = 0; i < dim; ++i)
			x[i] = dist(rng);
	}
	//*/

	// labels contain the ground truth (e.g. vertex classes) for each example (num_examples x 1).
	// Note that labels is not one-hot encoded vector and it can be computed
	// as y.argmax(axis=1) from one-hot encoded vector (y) of labels if required.
	size_t read_labels(std::string dataset_str, LabelList &labels) {
		std::cout << "Reading labels ... ";
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

	size_t read_features(std::string dataset_str, tensor_t &feats) {
		std::cout << "Reading features ... ";
		Timer t_read;
		t_read.Start();
		std::string filename = path + dataset_str + ".ft";
		std::ifstream in;
		std::string line;
		in.open(filename, std::ios::in);
		size_t m, n;
		in >> m >> n >> std::ws;
		assert(m == feats.size()); // m = number of vertices
		for (size_t i = 0; i < m; ++i) {
			feats[i].resize(n);
			for (size_t j = 0; j < n; ++j)
				feats[i][j] = 0;
		}
		while (std::getline(in, line)) {
			std::istringstream edge_stream(line);
			unsigned u, v;
			float_t w;
			edge_stream >> u;
			edge_stream >> v;
			edge_stream >> w;
			feats[u][v] = w;
		}
		/*
		for (size_t i = 0; i < 10; ++i)
			for (size_t j = 0; j < n; ++j)
				if (feats[i][j] > 0)
					std::cout << "feats[" << i << "][" << j << "]: " << feats[i][j] << std::endl;
		//*/
		in.close();
		t_read.Stop();
		std::cout << "Done, feature dimention: " << n << ", time: " << t_read.Millisecs() << " ms\n";
		return n;
	}

	unsigned read_graph(std::string dataset_str, Graph &graph) {
		//printf("Start readGraph\n");
		galois::StatTimer Tread("GraphReadingTime");
		Tread.start();
		LGraph lgraph;
		unsigned max_degree = 0;
		if (filetype == "el") {
			std::string filename = path + dataset_str + ".el";
			printf("Reading .el file: %s\n", filename.c_str());
			lgraph.read_edgelist(filename.c_str(), true); //symmetrize
			genGraph(lgraph, graph);
		} else if (filetype == "gr") {
			std::string filename = path + dataset_str + ".csgr";
			printf("Reading .gr file: %s\n", filename.c_str());
			galois::graphs::readGraph(graph, filename);
			/*
			galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const auto& vid) {
				graph.getData(vid) = 1;
				//for (auto e : graph.edges(n)) graph.getEdgeData(e) = 1;
			}, galois::chunk_size<256>(), galois::steal(), galois::loopname("assignVertexLabels"));
			std::vector<unsigned> degrees(graph.size());
			galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const auto& vid) {
				degrees[vid] = std::distance(graph.edge_begin(vid), graph.edge_end(vid));
			}, galois::loopname("computeMaxDegree"));
			max_degree = *(std::max_element(degrees.begin(), degrees.end()));
			*/
		} else { printf("Unkown file format\n"); exit(1); }
		if (filetype != "gr") {
			max_degree = lgraph.get_max_degree();
			lgraph.clean();
		}
		printf("max degree = %u\n", max_degree);
		Tread.stop();
		//printf("Done readGraph\n");
		std::cout << "num_vertices " << g.size() << " num_edges " << g.sizeEdges() << "\n";
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
				g.constructEdge(offset, lg.get_dest(offset), 0); // do not consider edge labels now
		}
	}

	inline acc_t masked_accuracy(size_t begin, size_t end, size_t count, MaskList &masks) {
		// comparing outputs with the ground truth (labels)
		//acc_t accuracy_all = 0.0;
		AccumF accuracy_all;
		accuracy_all.reset();
		//for (size_t i = begin; i < end; i++) {
		galois::do_all(galois::iterate(begin, end), [&](const auto& i) {
			if (masks[i] == 1) {
				int prediction = argmax(num_classes, layers[NUM_CONV_LAYERS-1]->next()->get_data()[i]);
				if ((label_t)prediction == labels[i]) accuracy_all += 1.0;
			}
		}, galois::chunk_size<256>(), galois::steal(), galois::loopname("getMaskedLoss"));
		//}
		return accuracy_all.reduce() / (acc_t)count;
	}
};

#endif
