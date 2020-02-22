#include "net.h"

void Net::init(std::string dataset_str, unsigned epochs, unsigned hidden1) {
	context = new Context();
	n = context->read_graph(dataset_str);
	num_classes = context->read_labels(dataset_str, n);
	context->degree_counting();
	context->norm_factor_counting(); // pre-compute normalizing factor
	num_epochs = epochs;
	std::cout << "Reading label masks ... ";
	train_mask.resize(n, 0);
	val_mask.resize(n, 0);
	if (dataset_str == "reddit") {
		train_begin = 0, train_count = 153431, train_end = train_begin + train_count;
		val_begin = 153431, val_count = 23831, val_end = val_begin + val_count;
		for (size_t i = train_begin; i < train_end; i++) train_mask[i] = 1;
		for (size_t i = val_begin; i < val_end; i++) val_mask[i] = 1;
	} else {
		train_count = read_masks(dataset_str, "train", train_begin, train_end, train_mask);
		val_count = read_masks(dataset_str, "val", val_begin, val_end, val_mask);
	}
	std::cout << "Done\n";

	num_layers = NUM_CONV_LAYERS + 1;
	feature_dims.resize(num_layers + 1);
	input_features.resize(n); // input embedding: N x D
	feature_dims[0] = read_features(dataset_str, input_features); // input feature dimension: D
	feature_dims[1] = hidden1; // hidden1 level embedding: 16
	feature_dims[2] = num_classes; // output embedding: E
	feature_dims[3] = num_classes; // normalized output embedding: E
	layers.resize(num_layers);
}

size_t Net::read_features(std::string dataset_str, tensor_t &feats) {
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
	in.close();
	t_read.Stop();
	std::cout << "Done, feature dimention: " << n << ", time: " << t_read.Millisecs() << " ms\n";
	return n;
}

void Net::train(optimizer *opt, bool need_validate) {
	std::cout << "\nStart training...\n";
	galois::StatTimer Tupdate("Train-WeightUpdate");
	galois::StatTimer Tfw("Train-Forward");
	galois::StatTimer Tbw("Train-Backward");
	galois::StatTimer Tval("Validation");
	Timer t_epoch;
	// run epoches
	for (unsigned i = 0; i < num_epochs; i++) {
		std::cout << "Epoch " << std::setw(2) << i << std::fixed << std::setprecision(3) << ":";
		t_epoch.Start();

		// training steps
		set_netphases(net_phase::train);
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
		set_netphases(net_phase::test);
		std::cout << " train_loss = " << std::setw(5) << train_loss << " train_acc = " << std::setw(5) << train_acc;
		t_epoch.Stop();
		double epoch_time = t_epoch.Millisecs();
		if (need_validate) {
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

