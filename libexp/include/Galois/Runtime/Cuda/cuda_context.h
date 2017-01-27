#pragma once
#include <cuda.h>
#include "gg.h"

struct CUDA_Context_Shared {
	unsigned int *num_nodes; // per host
	Shared<unsigned int> *nodes; // per host
	DeviceOnly<unsigned int> *offsets; // per host
  Shared<DynamicBitset> *is_updated; // per host
};

struct CUDA_Context_Common {
	int device;
	int id;
	unsigned int nowned;
	CSRGraphTy hg;
	CSRGraphTy gg;
  struct CUDA_Context_Shared master;
  struct CUDA_Context_Shared slave;
};

template<typename Type>
struct CUDA_Context_Field { 
  Shared<Type> data;
  DeviceOnly<Type> *master_data; // per host
  DeviceOnly<Type> *slave_data; // per host
  Shared<DynamicBitset> is_updated;
};

bool init_CUDA_context_common(struct CUDA_Context_Common *ctx, int device) {
	struct cudaDeviceProp dev;
	if(device == -1) {
		check_cuda(cudaGetDevice(&device));
	} else {
		int count;
		check_cuda(cudaGetDeviceCount(&count));
		if(device > count) {
			fprintf(stderr, "Error: Out-of-range GPU %d specified (%d total GPUs)", device, count);
			return false;
		}
		check_cuda(cudaSetDevice(device));
	}
	ctx->device = device;
	check_cuda(cudaGetDeviceProperties(&dev, device));
	fprintf(stderr, "%d: Using GPU %d: %s\n", ctx->id, device, dev.name);
	return true;
}

void load_graph_CUDA_common(struct CUDA_Context_Common *ctx, MarshalGraph &g, unsigned num_hosts) {
	CSRGraphTy &graph = ctx->hg;
	ctx->nowned = g.nowned;
	assert(ctx->id == g.id);
	graph.nnodes = g.nnodes;
	graph.nedges = g.nedges;
	if(!graph.allocOnHost(!g.edge_data)) {
		fprintf(stderr, "Unable to alloc space for graph!");
		exit(1);
	}
	memcpy(graph.row_start, g.row_start, sizeof(index_type) * (g.nnodes + 1));
	memcpy(graph.edge_dst, g.edge_dst, sizeof(index_type) * g.nedges);
	if(g.node_data) memcpy(graph.node_data, g.node_data, sizeof(node_data_type) * g.nnodes);
	if(g.edge_data) memcpy(graph.edge_data, g.edge_data, sizeof(edge_data_type) * g.nedges);
	ctx->master.num_nodes = (unsigned int *) calloc(num_hosts, sizeof(unsigned int));
	memcpy(ctx->master.num_nodes, g.num_master_nodes, sizeof(unsigned int) * num_hosts);
	ctx->master.nodes = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->master.offsets = (DeviceOnly<unsigned int> *) calloc(num_hosts, sizeof(DeviceOnly<unsigned int>));
	ctx->master.is_updated = (Shared<DynamicBitset> *) calloc(num_hosts, sizeof(Shared<DynamicBitset>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->master.num_nodes[h] > 0) {
			ctx->master.nodes[h].alloc(ctx->master.num_nodes[h]);
			ctx->master.offsets[h].alloc(ctx->master.num_nodes[h]);
			ctx->master.is_updated[h].alloc(1);
			ctx->master.is_updated[h].cpu_wr_ptr()->alloc(ctx->master.num_nodes[h]);
			memcpy(ctx->master.nodes[h].cpu_wr_ptr(), g.master_nodes[h], sizeof(unsigned int) * ctx->master.num_nodes[h]);
		}
	}
	ctx->slave.num_nodes = (unsigned int *) calloc(num_hosts, sizeof(unsigned int));
	memcpy(ctx->slave.num_nodes, g.num_slave_nodes, sizeof(unsigned int) * num_hosts);
	ctx->slave.nodes = (Shared<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
	ctx->slave.offsets = (DeviceOnly<unsigned int> *) calloc(num_hosts, sizeof(DeviceOnly<unsigned int>));
	ctx->slave.is_updated = (Shared<DynamicBitset> *) calloc(num_hosts, sizeof(Shared<DynamicBitset>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->slave.num_nodes[h] > 0) {
			ctx->slave.nodes[h].alloc(ctx->slave.num_nodes[h]);
			ctx->slave.offsets[h].alloc(ctx->slave.num_nodes[h]);
			ctx->slave.is_updated[h].alloc(1);
			ctx->slave.is_updated[h].cpu_wr_ptr()->alloc(ctx->slave.num_nodes[h]);
			memcpy(ctx->slave.nodes[h].cpu_wr_ptr(), g.slave_nodes[h], sizeof(unsigned int) * ctx->slave.num_nodes[h]);
		}
	}
	graph.copy_to_gpu(ctx->gg);
	printf("[%d] load_graph_GPU: %d owned nodes of total %d resident, %d edges\n", ctx->id, ctx->nowned, graph.nnodes, graph.nedges);
}

template<typename Type>
void load_graph_CUDA_field(struct CUDA_Context_Common *ctx, struct CUDA_Context_Field<Type> *field, unsigned num_hosts) {
	field->data.alloc(ctx->hg.nnodes);
	field->master_data = (DeviceOnly<Type> *) calloc(num_hosts, sizeof(DeviceOnly<Type>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->master.num_nodes[h] > 0) {
			field->master_data[h].alloc(ctx->master.num_nodes[h]);
		}
	}
	field->slave_data = (DeviceOnly<Type> *) calloc(num_hosts, sizeof(DeviceOnly<Type>));
	for(uint32_t h = 0; h < num_hosts; ++h){
		if (ctx->slave.num_nodes[h] > 0) {
			field->slave_data[h].alloc(ctx->slave.num_nodes[h]);
		}
	}
  field->is_updated.alloc(1);
  field->is_updated.cpu_wr_ptr()->alloc(ctx->hg.nnodes);
}

