typedef struct dPageRank {
   float value;
   unsigned int nout;
} PRResidual;
typedef void EdgeData;
typedef PRResidual NodeData;
#include "graph_header.h"
#define alpha (1.0 - 0.85)
__kernel void pagerank(__global int * graph_ptr) {
   int my_id = get_global_id(0);
   __local GraphType graph;
   initialize(&graph, graph_ptr);
   if(my_id < graph._num_nodes) {
      float sum = 0;
      __global NodeData * sdata = node_data(&graph, my_id);
      for(int i= out_neighbors_begin(&graph, my_id); i<out_neighbors_end(&graph, my_id); ++i) {
         int dst_id = out_neighbors(&graph, my_id, i);
         __global NodeData * dst_data = node_data(&graph, dst_id);
         sum+= dst_data->value / dst_data->nout;
      }//end for
      float v= (1.0 - alpha) * sum + alpha;
      float diff = fabs(v - sdata->value);
      sdata->value = v;
//      sdata->nout = 2;
   }//end if
}//end kernel


__kernel void pack_buffer(int num_items, __global int * p_node_data, __global int * p_bcast_ids, __global int * p_bcast_buffer){
   const int my_id = get_global_id(0);
   if(my_id < num_items){
      NodeData * node_data = (NodeData*)p_node_data;
      NodeData * bcast_buffer = (NodeData*)p_bcast_buffer;
      bcast_buffer[my_id].value = node_data[p_bcast_ids[my_id]].value;
      bcast_buffer[my_id].nout = node_data[p_bcast_ids[my_id]].nout;
   }

}
