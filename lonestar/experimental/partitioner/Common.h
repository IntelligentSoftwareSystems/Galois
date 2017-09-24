/*
 * Common.h
 *
 *  Created on: Jun 15, 2016
 *      Author: rashid
 */


#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include <set>
#include <vector>
#include <string>

#include "galois/graphs/FileGraph.h"
#include "galois/graphs/OfflineGraph.h"




#ifndef GDIST_EXP_APPS_PARTITIONER_COMMON_H_
#define GDIST_EXP_APPS_PARTITIONER_COMMON_H_


/******************************************************************
 *
 *****************************************************************/
std::string getPartitionFileName(std::string & basename, size_t hostID, size_t num_hosts) {
   std::string result = basename;
   result += ".PART.";
   result += std::to_string(hostID);
   result += ".OF.";
   result += std::to_string(num_hosts);
   return result;
}
std::string getMetaFileName(std::string & basename, size_t hostID, size_t num_hosts) {
   std::string result = basename;
   result += ".META.";
   result += std::to_string(hostID);
   result += ".OF.";
   result += std::to_string(num_hosts);
   return result;
}

std::string getReplicaInfoFileName(std::string & basename, size_t num_hosts) {
   std::string result = basename;
   result += ".REPLICA.FOR.";
   result += std::to_string(num_hosts);
   return result;
}
/******************************************************************
 *
 * To verify the partitioning -
 * 1) Read all the graphs individually.
 * 2) Read the partitioning information associated with each sub-graph.
 * 3) Go over each graph and verify that the edges are found in original graph
 *    and sum of edges equals the original graph.
 *****************************************************************/
struct NodeInfo {
   typedef size_t PartitionIDType;
   NodeInfo() :
         local_id(0), global_id(0), owner_id(0) {
   }
   NodeInfo(size_t l, size_t g, size_t o) :
         local_id(l), global_id(g), owner_id(o) {
   }
   size_t local_id;
   size_t global_id;
   PartitionIDType owner_id;
};
bool verifyParitions(std::string & basename, galois::graphs::OfflineGraph & g, size_t num_hosts) {
   bool verified = true;
   std::vector<galois::graphs::OfflineGraph*> pGraphs(num_hosts);
   std::vector<std::map<size_t, NodeInfo>> hostLocalToGlobalMap(num_hosts);
   std::cout << "Verifying partitions...\n";
   for (size_t h = 0; h < num_hosts; ++h) {
      std::string meta_file_name = getMetaFileName(basename, h, num_hosts);
      std::ifstream meta_file(meta_file_name, std::ifstream::binary);
      if (!meta_file.is_open()) {
         std::cout << "Unable to open file " << meta_file_name << "! Exiting!\n";
         return false;
      }
      size_t num_entries;
      meta_file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
      std::cout << "Partition :: " << h << " Number of nodes :: " << num_entries << "\n";
      for (size_t i = 0; i < num_entries; ++i) {
         std::pair<size_t, size_t> entry;
         size_t owner;
         meta_file.read(reinterpret_cast<char*>(&entry.first), sizeof(entry.first));
         meta_file.read(reinterpret_cast<char*>(&entry.second), sizeof(entry.second));
         meta_file.read(reinterpret_cast<char*>(&owner), sizeof(owner));
         hostLocalToGlobalMap[h][entry.second] = NodeInfo(entry.second, entry.first, owner);
      }

      std::string gFileName = getPartitionFileName(basename, h, num_hosts);
      pGraphs[h] = new galois::graphs::OfflineGraph(gFileName);
   }      //End for each host.

   std::vector<size_t> nodeOwners(g.size());
   for (auto & i : nodeOwners) {
      i = ~0;
   }
   std::vector<size_t> outEdgeCounts(g.size());
   std::vector<size_t> inEdgeCounts(g.size());
   for (size_t h = 0; h < num_hosts; ++h) {
      auto & graph = *pGraphs[h];
      std::cout << "Reading partition :: " << h << " w/ " << graph.size() << " nodes, and " << graph.sizeEdges() << " edges.\n";
      for (auto n = graph.begin(); n != graph.end(); ++n) {
         auto src = *n;
         assert(hostLocalToGlobalMap[h].find(src) != hostLocalToGlobalMap[h].end());
         auto g_src = hostLocalToGlobalMap[h][src].global_id;
         auto owner_src = hostLocalToGlobalMap[h][src].owner_id;
         if (nodeOwners[g_src] != ~0 && nodeOwners[g_src] != owner_src) {
            std::cout << "Error - Node:: " << g_src << " OwnerMismatch " << owner_src << " , " << nodeOwners[g_src] << "\n";
            verified = false;
         } else {
            nodeOwners[g_src] = owner_src;
         }
         for (auto e = graph.edge_begin(src); e != graph.edge_end(src); ++e) {
            auto dst = graph.getEdgeDst(e);
            auto g_dst = hostLocalToGlobalMap[h][dst].global_id;
            assert(hostLocalToGlobalMap[h].find(dst) != hostLocalToGlobalMap[h].end());
            outEdgeCounts[g_src]++;
            inEdgeCounts[g_dst]++;
            auto owner_dst = hostLocalToGlobalMap[h][dst].owner_id;
            if (nodeOwners[g_dst] != ~0 && nodeOwners[g_dst] != owner_dst) {
               std::cout << "Error - Node:: " << g_dst << " OwnerMismatch " << owner_dst << " , " << nodeOwners[g_dst] << "\n";
               verified = false;
            } else {
               nodeOwners[g_dst] = owner_dst;
            }
         }

      }
   }
   std::cout << "Matching against master copy:: " << g.size() << " nodes, and  " << g.sizeEdges() << " edges.\n";
   for (auto n = g.begin(); n != g.end(); ++n) {
      auto src = *n;
      for (auto e = g.edge_begin(*n); e != g.edge_end(*n); ++e) {
         outEdgeCounts[src]--;
         inEdgeCounts[g.getEdgeDst(e)]--;
      }      //End for-neighbors
   }      //End for-nodes

   std::cout << "Verification sizes :: In :: " << inEdgeCounts.size() << " , Out :: " << outEdgeCounts.size() << "\n";
   for (size_t i = 0; i < inEdgeCounts.size(); ++i) {
      if (inEdgeCounts[i] != 0) {
         std::cout << "Error - Node:: " << i << " inEdgeCount!=0, count=" << inEdgeCounts[i] << "\n";
         verified = false;
      };
      if (outEdgeCounts[i] != 0) {
         std::cout << "Error - Node:: " << i << " outEdgeCount!=0, count=" << outEdgeCounts[i] << "\n";
         verified = false;
      };
   }

   for (size_t h = 0; h < num_hosts; ++h) {
      delete pGraphs[h];
   }
   return verified;

}
/******************************************************************
 *
 *****************************************************************/


#endif /* GDIST_EXP_APPS_PARTITIONER_COMMON_H_ */
