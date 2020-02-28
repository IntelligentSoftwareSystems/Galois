#pragma once
#include <cassert>
#include <fstream>
#include <fcntl.h>
#include <cassert>
#include <unistd.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include "types.cuh"
#include "checker.h"

class CSRGraph {
protected:
	IndexT *row_start;
	IndexT *edge_dst;
	node_data_type *node_data;
	edge_data_type *edge_data;
	int *degrees;
	int nnodes;
	int nedges;
	bool device_graph;
public:
	CSRGraph() { init(); }
	//~CSRGraph() {}
	void init() {
		row_start = edge_dst = NULL;
		edge_data = NULL;
		node_data = NULL;
		nnodes = nedges = 0;
		device_graph = false;
	}
	int get_nnodes() { return nnodes; }
	int get_nedges() { return nedges; }
	void clean() {
		check_cuda(cudaFree(row_start));
		check_cuda(cudaFree(edge_dst));
	}
	__device__ __host__ bool valid_node(IndexT node) { return (node < nnodes); }
	__device__ __host__ bool valid_edge(IndexT edge) { return (edge < nedges); }
	__device__ __host__ IndexT getOutDegree(unsigned src) {
		assert(src < nnodes);
		return row_start[src+1] - row_start[src];
	};
	__device__ __host__ IndexT getDestination(unsigned src, unsigned edge) {
		assert(src < nnodes);
		assert(edge < getOutDegree(src));
		IndexT abs_edge = row_start[src] + edge;
		assert(abs_edge < nedges);
		return edge_dst[abs_edge];
	};
	__device__ __host__ IndexT getAbsDestination(unsigned abs_edge) {
		assert(abs_edge < nedges);
		return edge_dst[abs_edge];
	};
	inline __device__ __host__ IndexT getEdgeDst(unsigned edge) {
		assert(edge < nedges);
		return edge_dst[edge];
	};
	inline __device__ __host__ node_data_type getData(unsigned vid) {
		return node_data[vid];
	}
	inline __device__ __host__ IndexT edge_begin(unsigned src) {
		assert(src <= nnodes);
		return row_start[src];
	};
	inline __device__ __host__ IndexT edge_end(unsigned src) {
		assert(src <= nnodes);
		return row_start[src+1];
	};
	void read(std::string file, bool read_edge_data) {
		std::cout << "Reading graph fomr file: " << file << "\n";
		readFromGR(file.c_str(), read_edge_data);
	}
	void readFromGR(const char file[], bool read_edge_data) {
		std::ifstream cfile;
		cfile.open(file);
		// copied from GaloisCpp/trunk/src/FileGraph.h
		int masterFD = open(file, O_RDONLY);
		if (masterFD == -1) {
			printf("FileGraph::structureFromFile: unable to open %s.\n", file);
			return;
		}
		struct stat buf;
		int f = fstat(masterFD, &buf);
		if (f == -1) {
			printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
			abort();
		}
		size_t masterLength = buf.st_size;
		int _MAP_BASE = MAP_PRIVATE;
		void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
		if (m == MAP_FAILED) {
			m = 0;
			printf("FileGraph::structureFromFile: mmap failed.\n");
			abort();
		}
		//ggc::Timer t("graphreader");
		//t.start();
		//parse file
		uint64_t* fptr = (uint64_t*)m;
		__attribute__((unused)) uint64_t version = le64toh(*fptr++);
		assert(version == 1);
		uint64_t sizeEdgeTy = le64toh(*fptr++);
		uint64_t numNodes = le64toh(*fptr++);
		uint64_t numEdges = le64toh(*fptr++);
		uint64_t *outIdx = fptr;
		fptr += numNodes;
		uint32_t *fptr32 = (uint32_t*)fptr;
		uint32_t *outs = fptr32; 
		fptr32 += numEdges;
		if (numEdges % 2) fptr32 += 1;
		edge_data_type  *edgeData = (edge_data_type *)fptr32;
		
		// cuda.
		nnodes = numNodes;
		nedges = numEdges;
		printf("nnodes=%d, nedges=%d, sizeEdge=%d.\n", nnodes, nedges, sizeEdgeTy);
		allocOnHost(!read_edge_data);
		row_start[0] = 0;
		for (unsigned ii = 0; ii < nnodes; ++ii) {
			row_start[ii+1] = le64toh(outIdx[ii]);
			index_type degree = row_start[ii+1] - row_start[ii];
			for (unsigned jj = 0; jj < degree; ++jj) {
				unsigned edgeindex = row_start[ii] + jj;
				unsigned dst = le32toh(outs[edgeindex]);
				if (dst >= nnodes) printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, edgeindex);
				edge_dst[edgeindex] = dst;
				if(sizeEdgeTy && read_edge_data)
					edge_data[edgeindex] = edgeData[edgeindex];
			}
		}
		cfile.close();	// probably galois doesn't close its file due to mmap.
		//t.stop();
		//printf("read %lld bytes in %d ms (%0.2f MB/s)\n\r\n", masterLength, t.duration_ms(), (masterLength / 1000.0) / (t.duration_ms()));
		return;
	}

	void copy_to_gpu(struct CSRGraph &copygraph) {
		copygraph.nnodes = nnodes;
		copygraph.nedges = nedges;
		assert(copygraph.allocOnDevice(edge_data == NULL));
		check_cuda(cudaMemcpy(copygraph.edge_dst, edge_dst, nedges * sizeof(index_type), cudaMemcpyHostToDevice));
		if (edge_data != NULL) check_cuda(cudaMemcpy(copygraph.edge_data, edge_data, nedges * sizeof(edge_data_type), cudaMemcpyHostToDevice));
		check_cuda(cudaMemcpy(copygraph.node_data, node_data, nnodes * sizeof(node_data_type), cudaMemcpyHostToDevice));
		check_cuda(cudaMemcpy(copygraph.row_start, row_start, (nnodes+1) * sizeof(index_type), cudaMemcpyHostToDevice));
	}

	unsigned allocOnHost(bool no_edge_data) {
		assert(nnodes > 0);
		assert(!device_graph);
		if(row_start != NULL) return true;
		std::cout << "Allocating memory on CPU\n";
		size_t mem_usage = ((nnodes + 1) + nedges) * sizeof(index_type) + (nnodes) * sizeof(node_data_type);
		if (!no_edge_data) mem_usage += (nedges) * sizeof(edge_data_type);
		printf("Host memory for graph: %3u MB\n", mem_usage / 1048756);
		row_start = (index_type *) calloc(nnodes+1, sizeof(index_type));
		edge_dst  = (index_type *) calloc(nedges, sizeof(index_type));
		if (!no_edge_data) edge_data = (edge_data_type *) calloc(nedges, sizeof(edge_data_type));
		node_data = (node_data_type *) calloc(nnodes, sizeof(node_data_type));
		return ((no_edge_data || edge_data) && row_start && edge_dst && node_data);
	}

	unsigned allocOnDevice(bool no_edge_data) {
		if(edge_dst != NULL) return true;  
		assert(edge_dst == NULL); // make sure not already allocated
		check_cuda(cudaMalloc((void **) &edge_dst, nedges * sizeof(index_type)));
		check_cuda(cudaMalloc((void **) &row_start, (nnodes+1) * sizeof(index_type)));
		if (!no_edge_data) check_cuda(cudaMalloc((void **) &edge_data, nedges * sizeof(edge_data_type)));
		check_cuda(cudaMalloc((void **) &node_data, nnodes * sizeof(node_data_type)));
		device_graph = true;
		return (edge_dst && (no_edge_data || edge_data) && row_start && node_data);
	}
};
