// Graph Neural Networks
// Xuhao Chen <cxh@utexas.edu>
#include "lonestargnn.h"

const char* name = "Graph Convolutional Networks";
const char* desc = "Graph convolutional neural networks on an undirected graph";
const char* url  = 0;

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarGnnStart(argc, argv, name, desc, url);
	Net network; // the neural network to train
	network.init(dataset, epochs, hidden1);
	network.construct_layers(); // default setting for now; can be customized by the user
	network.print_layers_info();
	ResourceManager rm;

	// the optimizer used to update parameters, see optimizer.h for more details
	//optimizer *opt = new gradient_descent();
	//optimizer *opt = new adagrad(); 
	optimizer *opt = new adam();
	galois::StatTimer Ttrain("TrainAndVal");
	Ttrain.start();
	network.train(opt, do_validate); // do training using training samples
	Ttrain.stop();

	if (do_test) {
		// test using test samples
		size_t n = network.get_nnodes();
		acc_t test_loss = 0.0, test_acc = 0.0;
		size_t test_begin = 0, test_end = n, test_count = n;
		std::vector<mask_t> test_mask(n, 0);
		if (dataset == "reddit") {
			test_begin = 177262; test_count = 55703; test_end = test_begin + test_count;
			for (size_t i = test_begin; i < test_end; i++) test_mask[i] = 1;
		} else test_count = read_masks(dataset, "test", test_begin, test_end, test_mask);
		galois::StatTimer Ttest("Test");
		Ttest.start();
		double test_time = network.evaluate(test_begin, test_end, test_count, &test_mask[0], test_loss, test_acc);
		std::cout << "\nTesting: test_loss = " << test_loss << " test_acc = " << test_acc << " test_time = " << test_time << "\n";
		Ttest.stop();
	}
	std::cout << "\n" << rm.get_peak_memory() << "\n\n";
	return 0;
}

