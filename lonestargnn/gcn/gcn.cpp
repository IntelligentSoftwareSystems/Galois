// Graph Neural Networks
// Xuhao Chen <cxh@utexas.edu>
#include "lonestargnn.h"
#ifdef GALOIS_USE_DIST
#include "DistributedGraphLoader.h"
#endif

const char* name = "Graph Convolutional Networks";
const char* desc = "Graph convolutional neural networks on an undirected graph";
const char* url  = 0;

int main(int argc, char** argv) {
#ifndef GALOIS_USE_DIST
  galois::SharedMemSys G;
#else
  galois::DistMemSys G;
#endif
  LonestarGnnStart(argc, argv, name, desc, url);
  deepgalois::Net network; // the neural network to train

#ifdef GALOIS_USE_DIST
  std::vector<unsigned> dummyVec;
  Graph* dGraph = galois::graphs::constructSymmetricGraph<char, void>(dummyVec);
#endif

#ifndef GALOIS_USE_DIST
  // read network, features, ground truth, initialize metadata
  network.init(dataset, epochs, hidden1, add_selfloop, is_single_class);
#else
  network.init(dataset, epochs, hidden1, add_selfloop, dGraph);
#endif
  network.construct_layers(); // default setting for now; can be customized by
                              // the user
  network.print_layers_info();

  // tracks peak memory usage
  deepgalois::ResourceManager rm;

  // the optimizer used to update parameters, see optimizer.h for more details
  // optimizer *opt = new gradient_descent();
  // optimizer *opt = new adagrad();
  deepgalois::optimizer* opt = new deepgalois::adam();
  galois::StatTimer Ttrain("TrainAndVal");
  Ttrain.start();
  network.train(opt, do_validate); // do training using training samples
  Ttrain.stop();

  if (do_test) {
    // test using test samples
    galois::gPrint("\n");
#ifndef GALOIS_USE_DIST
    network.read_test_masks(dataset, NULL);
#else
    network.read_test_masks(dataset, dGraph);
#endif
    galois::StatTimer Ttest("Test");
    Ttest.start();
    acc_t test_loss = 0.0, test_acc = 0.0;
    double test_time = network.evaluate("test", test_loss, test_acc);
    galois::gPrint("Testing: test_loss = ", test_loss, " test_acc = ", test_acc,
                   " test_time = ", test_time, "\n");
    Ttest.stop();
  }
  galois::gPrint("\n", rm.get_peak_memory(), "\n\n");
  return 0;
}
