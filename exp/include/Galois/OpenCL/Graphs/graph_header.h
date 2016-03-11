//Graph header, created ::Tue Jun  2 16:02:26 2015

typedef struct NodeDataImpl{
   int dist;
}NodeData;
typedef int EdgeData;



typedef struct _GraphType { 
uint _num_nodes;
uint _num_edges;
 uint _node_data_size;
 uint _edge_data_size;
 __global NodeData *_node_data;
 __global uint *_out_index;
 __global uint *_out_neighbors;
 __global EdgeData *_out_edge_data;
 }GraphType;

uint edge_begin(__global GraphType * graph, uint node){
 return graph->_out_index[node];
}
uint edge_end(__global GraphType * graph, uint node){
 return graph->_out_index[node+1];
}
uint getEdgeDst(__global GraphType * graph,uint nbr){
 return graph->_out_neighbors[nbr];
}
__global EdgeData * getEdgeData(__global GraphType * graph,uint nbr){
 return &graph->_out_edge_data[nbr];
}
__global NodeData * getData(__global GraphType * graph, uint node){
 return &graph->_node_data[node];
}
void initialize(__global GraphType * graph, __global uint *mem_pool){
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
graph->_out_edge_data=(__global EdgeData*)&mem_pool[offset];
offset +=graph->_num_edges*graph->_edge_data_size;
}
