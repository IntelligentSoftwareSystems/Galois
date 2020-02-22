#ifndef _MODEL_H_
#define _MODEL_H_

#include <random>
#include "types.h"
#include "gtypes.h"
#include "context.h"
#include "galois/Timer.h"
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
	void init(std::string dataset_str, unsigned epochs, unsigned hidden1);
	size_t get_in_dim(size_t layer_id) { return feature_dims[layer_id]; }
	size_t get_out_dim(size_t layer_id) { return feature_dims[layer_id+1]; }
	size_t get_ft_dim() { return feature_dims[0]; }
	size_t read_features(std::string dataset_str, tensor_t &feats);
	void construct_layers() {
		std::cout << "\nConstructing layers...\n";
		append_conv_layer(0, true); // first conv layer
		append_conv_layer(1); // hidden1 layer
		append_out_layer(2); // output layer
		layers[0]->set_in_data(input_features); // feed input data
		set_contexts();
	}

	void set_contexts() {
		for (size_t i = 0; i < num_layers; i ++)
			layers[i]->set_context(context);
	}

	void set_netphases(net_phase phase) {
		for (size_t i = 0; i < num_layers; i ++)
			layers[i]->set_netphase(phase);
	}

	void print_layers_info() {
		for (size_t i = 0; i < num_layers; i ++)
			layers[i]->print_layer_info();
	}

	void append_conv_layer(size_t layer_id, bool act = false, bool norm = true, bool bias = false, bool dropout = true, float dropout_rate = 0.5) {
		assert(dropout_rate < 1.0);
		assert(layer_id < NUM_CONV_LAYERS);
		std::vector<size_t> in_dims(2), out_dims(2);
		in_dims[0] = out_dims[0] = n;
		in_dims[1] = get_in_dim(layer_id);
		out_dims[1] = get_out_dim(layer_id);
#ifdef CPU_ONLY
		layers[layer_id] = new graph_conv_layer(layer_id, act, norm, bias, dropout, dropout_rate, in_dims, out_dims);
#else
		layers[layer_id] = new graph_conv_layer(layer_id, act, norm, bias, dropout, dropout_rate, in_dims, out_dims);
#endif
		if(layer_id > 0) connect(layers[layer_id-1], layers[layer_id]);
	}

	void append_out_layer(size_t layer_id) {
		assert(layer_id > 0); // can not be the first layer
		std::vector<size_t> in_dims(2), out_dims(2);
		in_dims[0] = out_dims[0] = n;
		in_dims[1] = get_in_dim(layer_id);
		out_dims[1] = get_out_dim(layer_id);
		layers[layer_id] = new softmax_loss_layer(layer_id, in_dims, out_dims);
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
	void train(optimizer *opt, bool need_validate);
	size_t get_nnodes() { return n; }

protected:
	Context *context;
	size_t n; // number of samples: N
	size_t num_classes; // number of vertex classes: E
	size_t num_layers; // for now hard-coded: NUM_CONV_LAYERS + 1
	unsigned num_epochs; // number of epochs
	std::vector<size_t> feature_dims; // feature dimnesions for each layer
	tensor_t input_features; // input features: N x D
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

	// comparing outputs with the ground truth (labels)
	inline acc_t masked_accuracy(size_t begin, size_t end, size_t count, MaskList &masks) {
		AccumF accuracy_all;
		accuracy_all.reset();
		galois::do_all(galois::iterate(begin, end), [&](const auto& i) {
			if (masks[i] == 1) {
				int preds = argmax(num_classes, layers[NUM_CONV_LAYERS-1]->next()->get_data()[i]);
				if ((label_t)preds == context->get_label(i)) accuracy_all += 1.0;
			}
		}, galois::chunk_size<256>(), galois::steal(), galois::loopname("getMaskedLoss"));
		return accuracy_all.reduce() / (acc_t)count;
	}
};

#endif
