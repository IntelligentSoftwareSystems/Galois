#include "net.h"

void Net::init(std::string dataset_str, unsigned epochs, unsigned hidden1) {
	context = new Context();
	num_samples = context->read_graph(dataset_str);
	num_classes = context->read_labels(dataset_str);
	context->norm_factor_counting(); // pre-compute normalizing factor
	num_epochs = epochs;

	std::cout << "Reading label masks ... ";
	train_mask.resize(num_samples, 0);
	val_mask.resize(num_samples, 0);
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
	feature_dims[0] = context->read_features(dataset_str); // input feature dimension: D
	feature_dims[1] = hidden1; // hidden1 level embedding: 16
	feature_dims[2] = num_classes; // output embedding: E
	feature_dims[3] = num_classes; // normalized output embedding: E
	layers.resize(num_layers);
#ifndef CPU_ONLY
	context->copy_data_to_device(); // copy labels and input features to the device
#endif
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
		train_loss = fprop(train_begin, train_end, train_count, &train_mask[0]); // forward
		train_acc = masked_accuracy(train_begin, train_end, train_count, &train_mask[0]); // predict
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
			double val_time = evaluate(val_begin, val_end, val_count, &val_mask[0], val_loss, val_acc);
			Tval.stop();
			std::cout << " val_loss = " << std::setw(5) << val_loss << " val_acc = " << std::setw(5) << val_acc;
			std::cout << " time = " << epoch_time + val_time << " ms (train_time = " << epoch_time << " val_time = " << val_time << ")\n";
		} else {
			std::cout << " train_time = " << epoch_time << " ms\n";
		}
	}
}

void Net::construct_layers() {
	std::cout << "\nConstructing layers...\n";
	append_conv_layer(0, true); // first conv layer
	append_conv_layer(1); // hidden1 layer
	append_out_layer(2); // output layer
	layers[0]->set_in_data(context->get_in_ptr()); // feed input data
	set_contexts();
}

