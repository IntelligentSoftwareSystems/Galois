// Graph Neural Networks
// Xuhao Chen <cxh@utexas.edu>
#include "gnn.h"

const char* name = "GraphSage";
const char* desc = "A graph neural network variant: GraphSAGE";
const char* url  = 0;

class GraphSageMean : public graph_conv_layer {
  // user-defined combine function
};

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
  Net network;    // the neural network to train
  network.init(); // default setting for now; see its implementation to find how
                  // to customize it by the user
  ResourceManager rm;

  // the optimizer used to update parameters, see optimizer.h for more details
  // optimizer *opt = new gradient_descent();
  // optimizer *opt = new adagrad();
  optimizer* opt = new adam();
  galois::StatTimer Ttrain("Train");
  Ttrain.start();
  network.train(opt); // do training using training samples
  Ttrain.stop();

  // test using test samples
  acc_t test_loss = 0.0, test_acc = 0.0;
  size_t test_begin = 2312, test_end = 3312; // [2312, 3327) test size = 1015
                                             // TODO: replace ad-hoc settings
  galois::StatTimer Ttest("Test");
  Ttest.start();
  double test_time =
      network.evaluate(test_begin, test_end, test_loss, test_acc);
  std::cout << "\nTesting: test_loss = " << test_loss
            << " test_acc = " << test_acc << " test_time = " << test_time
            << "\n";
  Ttest.stop();

  std::cout << "\n" << rm.get_peak_memory() << "\n\n";
  return 0;
}
