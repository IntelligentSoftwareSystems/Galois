/*
 * pagerank_kernel.cl
 * OpenCL kernels for PageRank on heterogeneous Galois.
 *
 *  Created on: Jun 30, 2015
 *      Author: rashid
 */

#define alpha (1.0 - 0.85)

typedef struct dPageRank {
   float value;
   unsigned int nout;
} PRResidual;
typedef void EdgeData;
typedef PRResidual NodeData;
//#include "graph_header.h"
//##########################################
//Graph header, created ::Tue Jun  2 16:02:26 2015
typedef struct _GraphType {
   uint _num_nodes;
   uint _num_edges;
    uint _node_data_size;
    uint _edge_data_size;
    __global NodeData *_node_data;
    __global uint *_out_index;
    __global uint *_out_neighbors;
 }GraphType;
uint in_neighbors_begin(__local GraphType * graph, uint node){
 return 0;
}
uint in_neighbors_end(__local GraphType * graph, uint node){
 return graph->_out_index[node+1]-graph->_out_index[node];
}
uint in_neighbors_next(__local GraphType * graph, uint node){
 return 1;
}
uint in_neighbors(__local GraphType * graph, uint node, uint nbr){
 return graph->_out_neighbors[graph->_out_index[node]+nbr];
}
uint out_neighbors_begin(__local GraphType * graph, uint node){
 return 0;
}
uint out_neighbors_end(__local GraphType * graph, uint node){
 return graph->_out_index[node+1]-graph->_out_index[node];
}
uint out_neighbors_next(__local GraphType * graph, uint node){
 return 1;
}
uint out_neighbors(__local GraphType * graph,uint node,  uint nbr){
 return graph->_out_neighbors[graph->_out_index[node]+nbr];
}
__global NodeData * node_data(__local GraphType * graph, uint node){
 return &graph->_node_data[node];
}
void initialize(__local GraphType * graph, __global uint *mem_pool){
   uint offset =4;
   graph->_num_nodes=mem_pool[0];
   graph->_num_edges=mem_pool[1];
   graph->_node_data_size=mem_pool[2];
   graph->_edge_data_size=mem_pool[3];
   graph->_node_data= (__global NodeData *)&mem_pool[offset];
   offset +=graph->_num_nodes* graph->_node_data_size;
   graph->_out_index=&mem_pool[offset];
   offset +=graph->_num_nodes + 1;
   graph->_out_neighbors=&mem_pool[offset];
   offset +=graph->_num_edges;
}

//##########################################
__kernel void pagerank(__global int * graph_ptr,__global float* aux,  int num_items) {
   int my_id = get_global_id(0);
   __local GraphType graph;
   initialize(&graph, graph_ptr);
   if(my_id < num_items) {
      double sum = 0;
      __global NodeData * sdata = node_data(&graph, my_id);
//      printf("ND[%d->%d]", my_id,sdata->nout);
      for(int i= out_neighbors_begin(&graph, my_id); i<out_neighbors_end(&graph, my_id); ++i) {
         int dst_id = out_neighbors(&graph, my_id, i);
         __global NodeData * ddata = node_data(&graph, dst_id);
         sum+= ddata->value / ddata->nout;
//         printf("PR[%d->%d]", my_id,dst_id);
      }//end for
      float value= (1.0 - alpha) * sum + alpha;
      float diff = fabs(value - sdata->value);
//      sdata->value = value;
      aux[my_id] = value;
//      sdata->nout = 2;
   }//end if
}//end kernel

__kernel void writeback(__global int * graph_ptr,__global float * aux,  int num_items) {
   int my_id = get_global_id(0);
   __local GraphType graph;
   initialize(&graph, graph_ptr);
   if(my_id < num_items) {
      __global NodeData * sdata = node_data(&graph, my_id);
      sdata->value = aux[my_id];
//      sdata->nout = 2;
   }//end if
}//end kernel

__kernel void initialize_nout(__global int * graph_ptr, __global float* aux_array, int num_items) {
   int my_id = get_global_id(0);
   __local GraphType graph;
   initialize(&graph, graph_ptr);
   if(my_id < graph._num_nodes) {
      node_data(&graph, my_id)->value=1.0 - alpha;
      aux_array[my_id] = 1.0 - alpha;
      for(int i= out_neighbors_begin(&graph, my_id); i<out_neighbors_end(&graph, my_id); ++i) {
         int dst_id = out_neighbors(&graph, my_id, i);
         __global NodeData * dst_data = node_data(&graph, dst_id);
         atomic_add(&dst_data->nout,1);
//         printf("Init[%d->%d]", my_id,dst_id);
      }//end for
   }//end if
}//end kernel
