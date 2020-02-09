// Graph Neural Networks
// Xuhao Chen <cxh@utexas.edu>
#include "gnn.h"
#include "net.h"

const char* name = "Graph Convolutional Networks";
const char* desc = "Graph convolutional neural networks on an undirected graph";
const char* url  = 0;

class GraphSageMean: public Net {
	// user-defined aggregate function
	void aggregate(Graph *g, const FV2D &in, FV2D &out) {
		update_all(g, in, out);
	}

	// user-defined combine function
	void combine(const vec_t mat_v, const vec_t mat_u, const FV2D &fv_v, const FV2D &fv_u, FV2D &fv_out) {
		size_t dim = fv_out[0].size();
		galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
			FV a(dim, 0);
			FV b(dim, 0);
			mvmul(mat_v, fv_v[src], a);
			mvmul(mat_u, fv_u[src], b); 
			vadd(a, b, fv_out[src]); // out[src] = W*v + Q*u
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("combine"));
	}
};

class GCN: public Net {
public:
	// user-defined aggregate function
	void aggregate(Graph *g, const FV2D &in, FV2D &out) {
		update_all(g, in, out);
	}

	// user-defined combine function
	// matrix multiply feature vector
	void combine(const vec_t mat_v, const vec_t mat_u, const FV2D &fv_v, const FV2D &fv_u, FV2D &fv_out) {
		assert(mat_v.size() == fv_u[0].size);
		matmul(fv_u, mat_v, fv_out);
	}
};

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	GCN model;
	//GraphSageMean model;
	model.init();
	ResourceManager rm;

	// the optimizer used to update parameters
	//optimizer *opt = new adagrad(); 
	optimizer *opt = new adam();
	//optimizer *opt = new gradient_descent();
	galois::StatTimer Ttrain("Train");
	Ttrain.start();
	model.train(opt);
	Ttrain.stop();

	acc_t test_cost = 0.0, test_acc = 0.0;
	size_t n = model.get_nnodes();
	MaskList test_mask(n, 0);
	for (size_t i = 0; i < n; i++)
		if (i >= 2312) test_mask[i] = 1; // [2312, 3327) test size = 1015
	LabelList y_test;
	y_test.resize(n);
	for (size_t i = 0; i < n; i ++) y_test[i] = (test_mask[i] == 1 ? model.get_label(i) : -1);
	galois::StatTimer Ttest("Test");
	Ttest.start();
	double test_time = model.evaluate(y_test, test_mask, test_cost, test_acc);
	std::cout << "\nTesting: test_loss = " << test_cost << " test_acc = " << test_acc << " test_time = " << test_time << "\n";
	Ttest.stop();

	std::cout << "\n" << rm.get_peak_memory() << "\n\n";
	return 0;
}

