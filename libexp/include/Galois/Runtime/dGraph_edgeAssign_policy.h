
#include <boost/dynamic_bitset.hpp>
#include <stdlib.h>
//returns the host
#if 0
      bool is_assigned(uint64_t GID){
        if(node_mapping[GID] != ~0)
          return  true;
        else
          return false;
      }

      void assign_node(uint64_t GID, unsigned host){
        node_mapping[GID].push_back(host);
      }
#endif

uint32_t random_edge_assignment(uint64_t src, uint64_t dst, boost::dynamic_bitset<uint32_t>& src_bitset, boost::dynamic_bitset<uint32_t>& dst_bitset, uint32_t numHosts){

 uint32_t rand_number = rand() % numHosts;
 return rand_number;

}

uint32_t balanced_edge_assignment(uint64_t src, uint64_t dst, boost::dynamic_bitset<uint32_t>& src_bitset, boost::dynamic_bitset<uint32_t>& dst_bitset){


 boost::dynamic_bitset<uint32_t> intersection_set = src_bitset & dst_bitset;

 // not assigned to any host.
 if(intersection_set.none()){
  
 
 }
 return 0; 
  

}



#if 0
          /* POLICY */
          if(!is_assigned(src) && !is_assigned(dst)){
            auto src_host = src%base_hGraph::numHosts;
            auto dst_host = dst%base_hGraph::numHosts;

            if(num_edges_assigned[src_host].size() < num_edges_assigned[dst_host].size()){
              assign_node(src,src_host);
              assign_node(dst,src_host);
              num_edges_assigned[src_host] += 1;
            }
            else {
              assign_node(src,dst_host);
              assign_node(dst,dst_host);
              num_edges_assigned[dst_host] += 1;
            }
          }

          /***** atleast one of src or dst belongs to a host *****/
          else {
            if(is_assigned(src) && !is_assigned(dst)){
              auto src_host = get_assigned_host(src);
              auto dst_host = dst%base_hGraph::numHosts;
              if(num_edges_assigned[src_host].size() < num_edges_assigned[dst_host].size()){
                assign_node(dst, src_host);
              }
              else {
                assign_node(dst, dst_host);
                master_mapping[src_host] = src;
                slave_mapping[dst_host] = src;
              }
          
            if(!is_assigned(src) && is_assigned(dst)){
              auto src_host = src%base_hGraph::numHosts;
              auto dst_host = get_assigned_host(dst);
          
          
          }

        }
      }

#endif
