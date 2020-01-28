#include <random>
#include <iomanip>
#include <sys/time.h>
#include <sys/resource.h>
#include "gnn.h"
#include "lgraph.h"
#include "math_functions.hpp"

std::string path = "/h2/xchen/datasets/Learning/"; // path to the input dataset

class ResourceManager {
public:
	ResourceManager() {}
	~ResourceManager(){}
	//peak memory usage
	std::string get_peak_memory() {
		double kbm;
		struct rusage CurUsage;
		getrusage(RUSAGE_SELF, &CurUsage);
		kbm = (double)CurUsage.ru_maxrss;
		double mbm = kbm / 1024.0;
		double gbm = mbm / 1024.0;
		return
			"Peak memory: " +
			to_string_with_precision(mbm, 3) + " MB; " +
			to_string_with_precision(gbm, 3) + " GB";
	}
private:
	template <typename T = double>
	std::string to_string_with_precision(const T a_value, const int& n) {
		std::ostringstream out;
		out << std::fixed;
		out << std::setprecision(n) << a_value;
		return out.str();
	}
};

class Timer {
public:
	Timer() {}
	void Start() { gettimeofday(&start_time_, NULL); }
	void Stop() {
		gettimeofday(&elapsed_time_, NULL);
		elapsed_time_.tv_sec  -= start_time_.tv_sec;
		elapsed_time_.tv_usec -= start_time_.tv_usec;
	}
	double Seconds() const { return elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1e6; }
	double Millisecs() const { return 1000*elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1000; }
	double Microsecs() const { return 1e6*elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec; }
private:
	struct timeval start_time_;
	struct timeval elapsed_time_;
};

inline void init_matrix(size_t dim_x, size_t dim_y, FV2D &matrix) {
    // Glorot & Bengio (AISTATS 2010) init
	auto init_range = sqrt(6.0/(dim_x + dim_y));
	//std::cout << "Matrix init_range: (" << -init_range << ", " << init_range << ")\n";
	std::default_random_engine rng;
	std::uniform_real_distribution<FeatureT> dist(-init_range, init_range);
	matrix.resize(dim_x);
	for (size_t i = 0; i < dim_x; ++i) {
		matrix[i].resize(dim_y);
		for (size_t j = 0; j < dim_y; ++j)
			matrix[i][j] = dist(rng);
	}
	//for (size_t i = 0; i < 3; ++i)
	//	for (size_t j = 0; j < 3; ++j)
	//		std::cout << "matrix[" << i << "][" << j << "]: " << matrix[i][j] << std::endl;
}

inline void init_features(size_t dim, FV &x) {
	std::default_random_engine rng;
	std::uniform_real_distribution<FeatureT> dist(0, 0.1);
	for (size_t i = 0; i < dim; ++i)
		x[i] = dist(rng);
}

size_t read_labels(std::string dataset_str, LabelList &labels) {
	std::string filename = path + dataset_str + "-labels.txt";
	std::ifstream in;
	std::string line;
	in.open(filename, std::ios::in);
	size_t m, n;
	in >> m >> n >> std::ws;
	assert(m == labels.size()); // number of vertices
	std::cout << "label conuts: " << n << std::endl; // number of vertex classes
	IndexT v = 0;
	while (std::getline(in, line)) {
		std::istringstream label_stream(line);
		unsigned x;
		for (size_t idx = 0; idx < n; ++idx) {
			label_stream >> x;
			if (x != 0) {
				labels[v] = idx;
				break;
			}
		}
		v ++;
	}

	//for (size_t i = 0; i < 10; ++i)
	//	std::cout << "labels[" << i << "]: " << labels[i] << std::endl;
	return n;
}

size_t read_features(std::string dataset_str, FV2D &features) {
	std::string filename = path + dataset_str + ".ft";
	std::ifstream in;
	std::string line;
	in.open(filename, std::ios::in);
	size_t m, n;
	in >> m >> n >> std::ws;
	assert(m == features.size()); // m = number of vertices
	std::cout << "feature dimention: " << n << std::endl;
	for (size_t i = 0; i < m; ++i) {
		features[i].resize(n);
		for (size_t j = 0; j < n; ++j)
			features[i][j] = 0;
	}
	while (std::getline(in, line)) {
		std::istringstream edge_stream(line);
		IndexT u, v;
		FeatureT w;
		edge_stream >> u;
		edge_stream >> v;
		edge_stream >> w;
		features[u][v] = w;
	}
/*
	for (size_t i = 0; i < 10; ++i) {
		for (size_t j = 0; j < n; ++j) {
			if (features[i][j] > 0)
				std::cout << "features[" << i << "][" << j << "]: " << features[i][j] << std::endl;
		}
	}
//*/
	return n;
}

void genGraph(LGraph &lg, Graph &g) {
	g.allocateFrom(lg.num_vertices(), lg.num_edges());
	g.constructNodes();
	for (size_t i = 0; i < lg.num_vertices(); i++) {
		g.getData(i) = 1;
		auto row_begin = lg.get_offset(i);
		auto row_end = lg.get_offset(i+1);
		g.fixEndEdge(i, row_end);
		for (auto offset = row_begin; offset < row_end; offset ++)
			g.constructEdge(offset, lg.get_dest(offset), 0); // do not consider edge labels currently
	}
}

unsigned load_graph(Graph &graph, std::string filename, std::string filetype = "el") {
	LGraph lgraph;
	unsigned max_degree = 0;
	if (filetype == "el") {
		printf("Reading .el file: %s\n", filename.c_str());
		lgraph.read_edgelist(filename.c_str(), true); //symmetrize
		genGraph(lgraph, graph);
	} else if (filetype == "gr") {
		printf("Reading .gr file: %s\n", filename.c_str());
		galois::graphs::readGraph(graph, filename);
		galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const auto& vid) {
			graph.getData(vid) = 1;
			//for (auto e : graph.edges(n)) graph.getEdgeData(e) = 1;
		}, galois::chunk_size<256>(), galois::steal(), galois::loopname("assignVertexLabels"));
		std::vector<unsigned> degrees(graph.size());
		galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const auto& vid) {
			degrees[vid] = std::distance(graph.edge_begin(vid), graph.edge_end(vid));
		}, galois::loopname("computeMaxDegree"));
		max_degree = *(std::max_element(degrees.begin(), degrees.end()));
	} else { printf("Unkown file format\n"); exit(1); }
	if (filetype != "gr") {
		max_degree = lgraph.get_max_degree();
		lgraph.clean();
	}
	printf("max degree = %u\n", max_degree);
	return max_degree;
}

void read_graph(std::string dataset_str, Graph &g) {
	//printf("Start readGraph\n");
	galois::StatTimer Tread("GraphReadingTime");
	Tread.start();
	//std::string filename = dataset_str + ".gr";
	std::string filename = path + dataset_str + ".el";
	load_graph(g, filename);
	Tread.stop();
	//printf("Done readGraph\n");
	std::cout << "num_vertices " << g.size() << " num_edges " << g.sizeEdges() << "\n";
}

void set_masks(size_t n, MaskList &train_mask, MaskList &val_mask, MaskList &test_mask) {
	for (size_t i = 0; i < n; i++) {
		if (i < 120) train_mask[i] = 1; // [0, 120) train size = 120
		else if (i < 620) val_mask[i] = 1; // [120, 620) validation size = 500
		else if (i >= 2312) test_mask[i] = 1; // [2312, 3327) test size = 1015
		else ; // unlabeled vertices
	}
}

inline AccT masked_softmax_cross_entropy(FV2D &h_out, LabelList &labels, MaskList &masks) {
	size_t n = masks.size();
	std::vector<AccT> loss(n, 0.0);
	for (size_t i = 0; i < n; i++) loss[i] = 0.0;
	softmax_cross_entropy_with_logits(h_out, labels, loss);

	AccT sum_mask = std::accumulate(masks.begin(), masks.end(), (AccT)0);
	//float avg_mask = reduce_mean<MaskT>(masks);
	AccT avg_mask = sum_mask / (AccT)n;
	for (size_t i = 0; i < n; i ++) 
		loss[i] = loss[i] * (AccT)(masks[i]) / avg_mask;
	AccT sum_loss = std::accumulate(loss.begin(), loss.end(), (AccT)0);
	//AccT sum_loss = 0.0;
	//for (size_t i = 0; i < n; i ++) sum_loss += loss[i];
	return sum_loss / (AccT)n;
}

inline AccT masked_accuracy(LabelList &preds, LabelList &labels, MaskList &masks) {
	size_t n = labels.size();
	std::vector<AccT> accuracy_all(n, 0.0);
	for (size_t i = 0; i < n; i++)
		if (preds[i] == labels[i]) accuracy_all[i] = 1.0;
	auto sum_mask = std::accumulate(masks.begin(), masks.end(), (int)0);
	AccT avg_mask = (AccT)sum_mask / (AccT)n;
	for (size_t i = 0; i < n; i ++) 
		accuracy_all[i] = accuracy_all[i] * (AccT)masks[i] / avg_mask;
	AccT sum_accuracy_all = std::accumulate(accuracy_all.begin(), accuracy_all.end(), (AccT)0);
	return sum_accuracy_all / (AccT)n;
}

