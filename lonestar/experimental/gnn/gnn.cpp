// Graph Neural Networks
// Xuhao Chen <cxh@utexas.edu>
#include "gnn.h"
#include "model.h"

const char* name = "Graph Convolutional Networks";
const char* desc = "Graph convolutional neural networks on an undirected graph";
const char* url  = 0;

class GraphSageMean: public Model {
	// user-defined aggregate function
	void aggregate(const VertexID src, const FV2D &features, FV &sum) {
		unsigned degree = 0;
		for (const auto e : g.edges(src)) {
			const auto dst = g.getEdgeDst(e);
			vadd(sum, features[dst], sum);
			degree ++;
		}
		if (degree == 0) return;
		for (size_t i = 0; i < sum.size(); ++i) sum[i] /= degree; // average
	}

	// user-defined combine function
	void combine(const FV2D ma, const FV2D mb, const FV &a, const FV &b, FV &out) {
		size_t n = out.size();
		FV c(n, 0);
		FV d(n, 0);
		mvmul(ma, a, c);
		mvmul(mb, b, d);
		vadd(c, d, out);
	}
};

class GCN: public Model {
public:
	// user-defined aggregate function
	void aggregate(const VertexID src, const FV2D &features, FV &sum) {
		unsigned degree = 0;
		for (const auto e : g.edges(src)) {
			const auto dst = g.getEdgeDst(e);
			vadd(sum, features[dst], sum);
			degree ++;
		}
		vadd(sum, features[src], sum);
		degree ++;
		for (size_t i = 0; i < sum.size(); ++i) sum[i] /= degree; // average
	}

	// user-defined combine function
	// matrix multiply feature vector
	void combine(const FV2D matrix_v, const FV2D matrix_u, const FV &ft_v, const FV &fv_u, FV &out) {
		mvmul(matrix_v, fv_u, out);
	}
};

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	GCN model;
	//GraphSageMean model;
	model.init();
	ResourceManager rm;
	galois::StatTimer Ttrain("Train");
	Ttrain.start();
	model.train();
	Ttrain.stop();

	AccT test_cost = 0.0, test_acc = 0.0;
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

