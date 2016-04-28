//Graph header, created ::Tue Jun  2 16:02:26 2015
// Note that these have the graph pointer as the last argument
// This was done to work around the rewriter errors.
typedef struct _GraphType { 
 uint _num_nodes;
 uint _num_edges;
 uint _node_data_size;
 uint _edge_data_size;
 uint _num_owned;
 ulong _global_offset;
 uint _num_ghosts;
 __global NodeData *_node_data;
 __global uint *_out_index;
 __global uint *_out_neighbors;
 __global EdgeData *_out_edge_data;
 }GraphType;
typedef GraphType Graph;

uint edge_begin(uint node, __global GraphType * graph){
 return graph->_out_index[node];
}
uint edge_end(uint node, __global GraphType * graph){
 return graph->_out_index[node+1];
}
uint getEdgeDst(uint nbr, __global GraphType * graph){
 return graph->_out_neighbors[nbr];
}
__global EdgeData * getEdgeData(uint nbr, __global GraphType * graph){
 return &graph->_out_edge_data[nbr];
}
__global NodeData * getData(uint node, __global GraphType * graph){
 return &graph->_node_data[node];
}
uint getGID(uint lid, __global GraphType * graph){
   if (lid < graph->_num_owned){
      return lid + graph->_global_offset;
   }
   //TODO Finish implementation by adding ghost_map impl.
   return -1;
}
uint getLID(uint gid, __global GraphType * graph){
   //TODO RK - finish implementing. 
   return -1;
}

uint node_owner( uint lid, __global GraphType * graph){
//TODO RK - Finish implementing. 
return 0;
}

uint ghost_start(__global GraphType * graph){
   //TODO RK - Finish implementing/testing.
   return graph->_global_offset;
}


void initialize(__global GraphType * graph, __global uint *mem_pool){
uint offset =4;
graph->_num_nodes=mem_pool[0];
graph->_num_edges=mem_pool[1];
graph->_node_data_size=mem_pool[2];
graph->_edge_data_size=mem_pool[3];
graph->_num_owned=mem_pool[4];
graph->_global_offset=mem_pool[6];
graph->_node_data= (__global NodeData *)&mem_pool[offset];
offset +=graph->_num_nodes* graph->_node_data_size;
graph->_out_index=&mem_pool[offset];
offset +=graph->_num_nodes + 1;
graph->_out_neighbors=&mem_pool[offset];
offset +=graph->_num_edges;
graph->_out_edge_data=(__global EdgeData*)&mem_pool[offset];
offset +=graph->_num_edges*graph->_edge_data_size;
}

//For graphs with edge data.
__kernel void initialize_graph_struct(__global uint * res, __global uint * g_meta, __global NodeData *g_node_data, __global uint * g_out_index, __global uint * g_nbr, __global EdgeData * edge_data){
   __global GraphType * g = (__global GraphType *) res;
   g->_num_nodes = g_meta[0];
   g->_num_edges = g_meta[1];
   g->_node_data_size = g_meta[2];
   g->_edge_data_size= g_meta[3];
   g->_num_owned=g_meta[4];
   g->_global_offset=g_meta[6];
   g->_num_ghosts = g_meta[8];
   g->_node_data = g_node_data;
   g->_out_index= g_out_index;
   g->_out_neighbors = g_nbr;
   g->_out_edge_data = edge_data;
}


//For void graphs
__kernel void initialize_void_graph_struct(__global uint * res, __global uint * g_meta, __global NodeData *g_node_data, __global uint * g_out_index, __global uint * g_nbr){
   __global GraphType * g = (__global GraphType *) res;
   g->_num_nodes = g_meta[0];
   g->_num_edges = g_meta[1];
   g->_node_data_size = g_meta[2];
   g->_edge_data_size= g_meta[3];
   g->_num_owned=g_meta[4];
   g->_global_offset=g_meta[6];
   g->_node_data = g_node_data;
   g->_out_index= g_out_index;
   g->_out_neighbors = g_nbr;
}
