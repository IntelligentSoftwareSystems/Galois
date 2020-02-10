// Graph Neural Networks
// Xuhao Chen <cxh@utexas.edu>
#include "gnn.h"

const char* name = "Graph Convolutional Networks";
const char* desc = "Graph convolutional neural networks on an undirected graph";
const char* url  = 0;

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Net network; // the neural network to train
	network.init(); // default setting for now; see its implementation to find how to customize it by the user
	ResourceManager rm;

	// the optimizer used to update parameters, see optimizer.h for more details
	//optimizer *opt = new gradient_descent();
	//optimizer *opt = new adagrad(); 
	optimizer *opt = new adam();
	galois::StatTimer Ttrain("Train");
	Ttrain.start();
	network.train(opt); // do training using training samples
	Ttrain.stop();

	// test using test samples
	size_t n = network.get_nnodes();
	acc_t test_loss = 0.0, test_acc = 0.0;
	size_t test_begin = 2312, test_end = n; // [2312, 3327) test size = 1015 TODO: replace ad-hoc settings
	MaskList test_mask(n, 0);
	size_t test_sample_count = read_masks(dataset, "test", test_begin, test_end, test_mask);
	galois::StatTimer Ttest("Test");
	Ttest.start();
	double test_time = network.evaluate(test_begin, test_end, test_sample_count, test_mask, test_loss, test_acc);
	std::cout << "\nTesting: test_loss = " << test_loss << " test_acc = " << test_acc << " test_time = " << test_time << "\n";
	Ttest.stop();

	std::cout << "\n" << rm.get_peak_memory() << "\n\n";
	return 0;
}

