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
  network.init(dataset, epochs, hidden1, add_selfloop);
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
    galois::gPrint("\n");
    // test using test samples
    size_t n        = network.get_nnodes();
    acc_t test_loss = 0.0, test_acc = 0.0;
    size_t test_begin = 0, test_end = n, test_count = n;
    std::vector<mask_t> test_mask(n, 0);
    if (dataset == "reddit") {
      test_begin = 177262;
      test_count = 55703;
      test_end   = test_begin + test_count;
#ifndef GALOIS_USE_DIST
      for (size_t i = test_begin; i < test_end; i++)
        test_mask[i] = 1;
#else
      for (size_t i = test_begin; i < test_end; i++)  {
        if (dGraph->isLocal(i)) {
          test_mask[dGraph->getLID(i)] = 1;
        }
      }
#endif
    } else {
#ifndef GALOIS_USE_DIST
      test_count = deepgalois::read_masks(dataset, "test", test_begin, test_end, test_mask);
#else
      test_count = deepgalois::read_masks(dataset, "test", test_begin, test_end,
                                          test_mask, dGraph);
#endif
    }
    galois::StatTimer Ttest("Test");
    Ttest.start();
    double test_time = network.evaluate(test_begin, test_end, test_count,
                                        &test_mask[0], test_loss, test_acc);
    galois::gPrint("Testing: test_loss = ", test_loss, " test_acc = ", test_acc,
                   " test_time = ", test_time, "\n");
    Ttest.stop();
  }
  galois::gPrint("\n", rm.get_peak_memory(), "\n\n");
  return 0;
}
