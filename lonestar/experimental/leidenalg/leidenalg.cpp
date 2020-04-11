/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "galois/AtomicHelpers.h"
#include "galois/AtomicWrapper.h"

#include <iostream>
#include <fstream>
#include <deque>
#include <type_traits>

#include <random>
#include <math.h>
#include <algorithm>

#include "Lonestar/BoilerPlate.h"
#include "leidenalg.h"
#include "galois/DynamicBitset.h"

namespace cll = llvm::cl;

static const char* name = "Louvain Clustering";

static const char* desc =
    "Cluster nodes of the graph using Louvain Clustering";

static const char* url = "louvain_clustering";

enum Algo {
  coloring,
  foreach,
  delay,
  doall
};

enum Quality{
	CPM,
	Mod
};


static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);

static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumValN(Algo::coloring, "Coloring", "Using colors to mitigate conflicts"),
                clEnumValN(Algo::foreach, "Foreach", "Using galois for_each for conflict mitigation"),
                clEnumValN(Algo::delay, "Delay", "Using galois for_each for conflict mitigation but delay the updation"),
                clEnumValN(Algo::doall, "Doall", "Using galois for_each for conflict mitigation"),
                clEnumValEnd),
    cll::init(Algo::foreach));

static cll::opt<Quality> quality(
    "quality", cll::desc("Choose an option:"),
    cll::values(clEnumValN(Quality::CPM, "CPM", "Using CPM Quality"),
                clEnumValN(Quality::Mod, "Mod", "Using mod"),
                clEnumValEnd),
    cll::init(Quality::Mod));

static cll::opt<bool> enable_VF("enable_VF",
  cll::desc("Flag to enable vertex following optimization."),
  cll::init(false));

static cll::opt<double> c_threshold("c_threshold",
  cll::desc("Threshold for modularity gain"),
  cll::init(0.01));

static cll::opt<double> threshold("threshold",
  cll::desc("Total threshold for modularity gain"),
  cll::init(0.01));

static cll::opt<uint32_t> max_iter("max_iter",
  cll::desc("Maximum number of iterations to execute"),
  cll::init(10));

static cll::opt<bool> output_CID("output_CID",
  cll::desc("Flag to enable cluster ID printing."),
  cll::init(false));

static cll::opt<std::string> output_CID_filename("output_CID_filename",
  cll::desc("File name to output cluster IDs."),
  cll::init("output_CID_filename"));

int64_t maxQualityWithoutSwaps(Graph &graph, uint64_t degree_wt, int64_t curr_comm_ass,
 std::map<uint64_t, uint64_t>& cluster_local_map,
 std::vector<uint64_t>& counter, uint64_t self_loop_wt, CommArray &c_info, uint64_t flatSize){

 	int64_t local_target = -1;

		switch(quality){
          case CPM:
            local_target = maxCPMQualityWithoutSwaps(cluster_local_map, counter, self_loop_wt, c_info, degree_wt, curr_comm_ass, flatSize);
           break;
          case Mod:
//            local_target = maxModularityWithoutSwaps(cluster_local_map, counter, self_loop_wt, c_info, degree_wt, curr_comm_ass);
  						local_target = maxModularity(cluster_local_map, counter, self_loop_wt, c_info, degree_wt, curr_comm_ass);
	          break;
         	default:
            std::abort();
        }
	return local_target;
}

void moveNodesFast(Graph &graph, CommArray &c_info){

	galois::InsertBag<GNode> bag_curr;
	galois::InsertBag<GNode> bag_next;

	galois::LargeArray<bool> inBag;

	inBag.allocateBlocked(graph.size());

	galois::do_all(galois::iterate(graph),
	[&](GNode n){

		bag_curr.push(n);
		inBag[n] = true;
	}, galois::steal());
	
	galois::do_all(galois::iterate((uint32_t)0, (uint32_t)(2*graph.size()+1)),
    [&](uint32_t n){

			c_info[n].size = (uint64_t) 0;
			c_info[n].degree_wt = (uint64_t) 0;
			c_info[n].flatSize = (uint64_t) 0; 
	}, galois::steal());

	galois::do_all(galois::iterate(graph),
	[&] (GNode n){
		
		galois::atomicAdd(c_info[graph.getData(n).curr_comm_ass].size, (uint64_t) 1);
		galois::atomicAdd(c_info[graph.getData(n).curr_comm_ass].degree_wt, graph.getData(n).degree_wt);
		galois::atomicAdd(c_info[graph.getData(n).curr_comm_ass].flatSize, graph.getData(n).flatSize);
	}, galois::steal());


	while(true){

		galois::do_all(galois::iterate(bag_curr),
			[&] (GNode n){
			
        auto& n_data = graph.getData(n,flag_write_lock);
				inBag[n] = false;

				//uint64_t degree = std::distance(graph.edge_begin(n, flag_no_lock), graph.edge_end(n, flag_no_lock));

				std::map<uint64_t, uint64_t> cluster_local_map;
				std::vector<uint64_t> counter;			
			 	uint64_t num_unique_clusters = 1;
 			 	uint64_t self_loop_wt = 0;

//  if(degree > 0){

    		cluster_local_map[n_data.curr_comm_ass] = 0; // Add n's current cluster
    		counter.push_back(0); //Initialize the counter to zero (no edges incident yet)

             
    		for(auto ii = graph.edge_begin(n, flag_no_lock); ii != graph.edge_end(n, flag_no_lock); ++ii) {
    	
					GNode dst = graph.getEdgeDst(ii);
    
      		auto edge_wt = graph.getEdgeData(ii, flag_no_lock);             
	   	
					if(dst == n){
      			self_loop_wt += edge_wt; // Self loop weights is recorded
     			}
                        
					auto stored_already = cluster_local_map.find(graph.getData(dst).curr_comm_ass); // Check if it already exists
                        
					if(stored_already != cluster_local_map.end()) {
      			counter[stored_already->second] += edge_wt;
      		} else {
                        
			 			cluster_local_map[graph.getData(dst).curr_comm_ass] = num_unique_clusters;
       			counter.push_back(edge_wt);
       			num_unique_clusters++;
      		}
    		} // End edge loop

				if(n_data.curr_comm_ass != graph.size()+n){
	
					cluster_local_map[graph.size()+n] = num_unique_clusters;
					counter.push_back(0);
					num_unique_clusters++;
				}
		
				//create cluster local map 

				int64_t local_target= maxQualityWithoutSwaps(graph, n_data.degree_wt, n_data.curr_comm_ass, cluster_local_map, counter, self_loop_wt, c_info, n_data.flatSize);

		//		std::cout << "moving to" << local_target << "from " << n_data.curr_comm_ass << std::endl;				
//				auto& n_data = graph.getData(n, flag_write_lock);

				if(local_target != -1 && local_target != graph.getData(n).curr_comm_ass){
	
					galois::atomicAdd(c_info[local_target].degree_wt, n_data.degree_wt);
          galois::atomicAdd(c_info[local_target].size, (uint64_t)1);
					galois::atomicAdd(c_info[local_target].flatSize, n_data.flatSize);

          galois::atomicSubtract(c_info[n_data.curr_comm_ass].degree_wt, n_data.degree_wt);
          galois::atomicSubtract(c_info[n_data.curr_comm_ass].size, (uint64_t)1);
					galois::atomicSubtract(c_info[n_data.curr_comm_ass].flatSize, n_data.flatSize);

					n_data.curr_comm_ass = local_target;

					//explore neighbors and add them to queue
					for(auto e: graph.edges(n)){
			
						GNode u = graph.getEdgeDst(e);
						if(graph.getData(u).curr_comm_ass != local_target && !inBag[u]){					
								inBag[u] = true;
							}
						}
					}

			}, galois::steal() );

			//clear the queue
			bag_curr.clear();

			//populate bag_next
			galois::do_all(galois::iterate(graph),
			[&](GNode n){
			
					if(inBag[n])
						bag_curr.push(n);
				}, galois::steal());
	
			//breaking criterion
			if(bag_curr.begin() == bag_curr.end())
				break;

	}//end while

	inBag.destroy();
	inBag.deallocate();
}


//node n should have updated degree_wt values
//subcomm_info should have updated size values and external edge weights
uint64_t getRandomSubcommunity(Graph& graph, uint64_t n, CommArray &subcomm_info, uint64_t flatSize_comm, uint64_t degree_comm){

  uint64_t rand_subcomm = -1;
  uint64_t curr_subcomm = graph.getData(n).curr_subcomm_ass;

  std::map<uint64_t, uint64_t> cluster_local_map; // Map each neighbor's subcommunity to local number: Subcommunity --> Index
	std::vector<uint64_t> counter; //Number of edges to each unique subcommunity

	uint64_t num_unique_clusters = 1;

  cluster_local_map[curr_subcomm] = 0; // Add n's current subcommunity
	counter.push_back(0); //Initialize the counter to zero (no edges incident yet)
	
	uint64_t self_loop_wt = 0;
	uint64_t degree_wt = graph.getData(n).degree_wt;

  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
    GNode dst = graph.getEdgeDst(ii);
    auto edge_wt = graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded

		if(graph.getData(dst).curr_comm_ass != graph.getData(n).curr_comm_ass)
			continue;

		if(dst == n){
      self_loop_wt += edge_wt; // Self loop weights is recorded
		}
    auto stored_already = cluster_local_map.find(graph.getData(dst).curr_subcomm_ass); // Check if it already exists

		if(stored_already != cluster_local_map.end()) {
      counter[stored_already->second] += edge_wt;
    }
    else {
      cluster_local_map[graph.getData(dst).curr_subcomm_ass] = num_unique_clusters;
      counter.push_back(edge_wt);
      num_unique_clusters++;
    }
  } // End edge loop	

	std::map<uint64_t, uint64_t> new_cluster_local_map;
  std::vector<double> prefix_transformed_quality_increment;
  num_unique_clusters = 0;
  double total = 0.0f;

	double max_increment = -1;

	int64_t idx = -1;
	int64_t max_idx = -1;

  for(auto pair: cluster_local_map){

    auto subcomm = pair.first;
    if(curr_subcomm == subcomm)
      continue;

    double flatSize_subcomm = (double) subcomm_info[subcomm].flatSize;

    //check if subcommunity is well connected
//    if(subcomm_info[subcomm].external_edge_wt < resolution*flatSize_subcomm*((double)flatSize_comm - flatSize_subcomm))
  //    continue;


		double degree_subcomm = (double) subcomm_info[subcomm].degree_wt.load();

		if(subcomm_info[subcomm].external_edge_wt < constant*degree_subcomm*((double)degree_comm - degree_subcomm))
      continue;

    double quality_increment = 0;
	
		switch(quality){
          case CPM:
						quality_increment = diffCPMQuality(curr_subcomm, subcomm, cluster_local_map, counter, subcomm_info, self_loop_wt, graph.getData(n).flatSize);
						break;
					case Mod:
						quality_increment = diffModQuality(curr_subcomm, subcomm, cluster_local_map, counter, subcomm_info, self_loop_wt, degree_wt);
						break;
					default:
            std::abort();
		} 

    if(quality_increment >= -0.00000001f && quality_increment < 0)
			quality_increment = 0.0f;

		if(quality_increment >= 0){

      new_cluster_local_map[num_unique_clusters] = subcomm;
      double transformed_quality_increment = (double) fastExp(quality_increment/randomness);
	  
			total += transformed_quality_increment;
      prefix_transformed_quality_increment.push_back(total);
      num_unique_clusters++;
 
			if(max_increment < quality_increment || ((max_increment == quality_increment) && (subcomm > max_idx))){
				max_increment  = quality_increment;
				max_idx = subcomm;
			}
   	}
  }

//	std::cout << "tota: " << total << std::endl;
//	return idx;
	if(isinf(total))
		return max_idx ;

//	std::cout << "tota: " << total << std::endl;
  long double r = distribution(generator);
  r = total*r;

  int64_t min_idx = -1;
  max_idx = num_unique_clusters;
	int64_t mid_idx;

  while(min_idx < max_idx -1){

    mid_idx = (min_idx + max_idx)/2;

    if(prefix_transformed_quality_increment[mid_idx] >= r)
      max_idx = mid_idx;
    else
      min_idx = mid_idx;
  }

  if(max_idx < num_unique_clusters)
    return new_cluster_local_map[max_idx];
  else
    return -1;
}


//this should be implemented in serial; since this is called in parallel for all communities
void mergeNodesSubset(Graph &graph, std::vector<GNode> &S, int64_t comm_id, uint64_t comm_flatSize, uint64_t comm_degree, CommArray& subcomm_info){

	//select set R
	std::vector<GNode> R;
	
	for(auto n: S){
		
		uint64_t total = 0, internal = 0;
		for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                        
			GNode dst = graph.getEdgeDst(ii);
      auto edge_wt = graph.getEdgeData(ii); // Self loop weights is recorded
      if(dst != n && graph.getData(dst).curr_comm_ass == comm_id){
      	total += edge_wt;
			}
			if(graph.getData(dst).curr_comm_ass == comm_id)
				internal += edge_wt;
		}

		graph.getData(n).internal_degree_wt = internal;

		double flatSize_n = (double)graph.getData(n).flatSize;
		double degree_n = (double) graph.getData(n).degree_wt; 

	//	if(total >= resolution*flatSize_n*((double)comm_flatSize - flatSize_n))
		if(total >= constant*degree_n*((double) comm_degree - degree_n))
			R.push_back(n);	

		subcomm_info[n].flatSize = graph.getData(n).flatSize;
		subcomm_info[n].external_edge_wt = total;
		subcomm_info[n].size = (uint64_t) 1;
		subcomm_info[n].degree_wt = graph.getData(n).degree_wt;
	}

	for(auto n:R){
	
		if(subcomm_info[graph.getData(n).curr_subcomm_ass].size == (uint64_t) 1){
		
			int subcomm_ass = getRandomSubcommunity(graph, n, subcomm_info, comm_flatSize, comm_degree);

			if(subcomm_ass != -1 && subcomm_ass != graph.getData(n).curr_subcomm_ass){
	
				graph.getData(n).curr_subcomm_ass = subcomm_ass;

				//update Subcomm info
				subcomm_info[subcomm_ass].flatSize += graph.getData(n).flatSize;
				subcomm_info[subcomm_ass].size += (uint64_t) 1;
				subcomm_info[subcomm_ass].degree_wt += graph.getData(n).degree_wt;
	
				for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {

      		GNode dst = graph.getEdgeDst(ii);
      		auto edge_wt = graph.getEdgeData(ii);	
				
					if(dst != n && graph.getData(dst).curr_subcomm_ass == subcomm_ass){
						subcomm_info[subcomm_ass].external_edge_wt -= edge_wt;	
					}
					else if (dst != n && graph.getData(dst).curr_comm_ass == comm_id){
						subcomm_info[subcomm_ass].external_edge_wt += edge_wt;
					}
				}
			}	
		}
	}
}

void refinePartition(Graph &graph){

	//set singleton subcommunities
	galois::do_all(galois::iterate(graph),
		[&] (GNode n){
		
			graph.getData(n).curr_subcomm_ass = n;
		}, galois::steal());


	//populate nodes into communities
	std::vector<std::vector<GNode>> myVec(2*graph.size()+1);

	CommArray comm_info;

	comm_info.allocateBlocked(2*graph.size()+1);


	galois::do_all(galois::iterate((uint32_t)0, (uint32_t)(2*graph.size()+1)),
    [&](uint32_t n){

			comm_info[n].flatSize = (uint64_t) 0;
			comm_info[n].degree_wt = (uint64_t) 0;
		}, galois::steal());


	for(auto n: graph){

		myVec[graph.getData(n).curr_comm_ass].push_back(n);
	}

	galois::do_all(galois::iterate(graph),
	[&] (GNode n){

		galois::atomicAdd(comm_info[graph.getData(n).curr_comm_ass].flatSize, graph.getData(n).flatSize);
		galois::atomicAdd(comm_info[graph.getData(n).curr_comm_ass].degree_wt, graph.getData(n).degree_wt);
	}, galois::steal());
	
 
	CommArray subcomm_info;
	
	subcomm_info.allocateBlocked(graph.size()+1);
	
	//call mergeNodesSubset for each community in parallel	
	galois::do_all(galois::iterate((uint32_t)0, (uint32_t)(2*graph.size()+1)),
  	[&](uint32_t c){
		
			if(myVec[c].size() > 1){
				mergeNodesSubset(graph, myVec[c], c, comm_info[c].flatSize, comm_info[c].degree_wt, subcomm_info);
			}
		}, galois::steal());

	comm_info.destroy();
	comm_info.deallocate();

	subcomm_info.destroy();
  subcomm_info.deallocate();
}


//num_unique_clusters must be the total number of subcommunities
void buildNextLevelGraph(Graph& graph, Graph& graph_next, uint64_t num_unique_clusters) {
  std::cerr << "Inside buildNextLevelGraph\n";

  galois::StatTimer TimerGraphBuild("Timer_Graph_build");
  TimerGraphBuild.start();
  uint32_t num_nodes_next = num_unique_clusters;
  uint64_t num_edges_next = 0; //Unknown right now


  //galois::LargeArray<galois::InsertBag<uint64_t>> cluster_bags;
  //cluster_bags.allocateBlocked(num_unique_clusters);
  //std::vector<galois::InsertBag<uint64_t>> cluster_bags(num_unique_clusters);
  std::vector<std::vector<GNode>> cluster_bags(num_unique_clusters);
#if 0
  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                    auto n_data = graph.getData(n, flag_no_lock);
                    cluster_bags[n_data.curr_comm_ass].push_back(n);
                },
                galois::loopname("Cluster Bags"),
                galois::steal());
#endif

	std::vector<uint64_t> cluster_flatsize(num_unique_clusters);
	std::vector<int64_t> comm_ass(num_unique_clusters);

	//initialize to 0
	for(int i=0;i<num_unique_clusters;i++)
		cluster_flatsize[i] = (uint64_t) 0;
	
	
  // Comment: Serial separation is better than do_all due to contention
  for(GNode n = 0; n < graph.size(); ++n) {
      auto n_data = graph.getData(n, flag_no_lock);
      if(n_data.curr_subcomm_ass != -1){
        cluster_bags[n_data.curr_subcomm_ass].push_back(n);
				cluster_flatsize[n_data.curr_subcomm_ass] += n_data.flatSize;
				if(n_data.curr_comm_ass < graph.size())
					comm_ass[n_data.curr_subcomm_ass] = graph.getData(n_data.curr_comm_ass).curr_subcomm_ass;
				else
					comm_ass[n_data.curr_subcomm_ass] = n_data.curr_subcomm_ass;
			}
      //else galois::gPrint("ISOLATED NODE : ", n, "\n");
  }


  std::vector<std::vector<uint32_t>> edges_id(num_unique_clusters);
  std::vector<std::vector<EdgeTy>> edges_data(num_unique_clusters);

  /* First pass to find the number of edges */
  galois::do_all(galois::iterate((uint64_t)0, num_unique_clusters),
                [&](uint64_t c) {
                    std::map<uint64_t, uint64_t> cluster_local_map;
                    uint64_t num_unique_clusters = 0;
                    for(auto cb_ii = cluster_bags[c].begin(); cb_ii != cluster_bags[c].end(); ++cb_ii) {

                      assert(graph.getData(*cb_ii, flag_no_lock).curr_subcomm_ass == c); // All nodes in this bag must have same cluster id

                      for(auto ii = graph.edge_begin(*cb_ii); ii != graph.edge_end(*cb_ii); ++ii) {
                        GNode dst = graph.getEdgeDst(ii);
                        auto dst_data = graph.getData(dst, flag_no_lock);
                       	assert(dst_data.curr_subcomm_ass <=num_unique_clusters );
                        assert(dst_data.curr_subcomm_ass !=  -1);
                        auto stored_already = cluster_local_map.find(dst_data.curr_subcomm_ass); // Check if it already exists
                        if(stored_already != cluster_local_map.end()) {
                          edges_data[c][stored_already->second] += graph.getEdgeData(ii);
                        } else {
                          cluster_local_map[dst_data.curr_subcomm_ass] = num_unique_clusters;
                          edges_id[c].push_back(dst_data.curr_subcomm_ass);
                          edges_data[c].push_back(graph.getEdgeData(ii));
                          num_unique_clusters++;
                        }
                      } // End edge loop
                    }
                }, galois::steal(),
                   galois::loopname("BuildGrah: Find edges"));

  /* Serial loop to reduce all the edge counts */
  std::vector<uint64_t> prefix_edges_count(num_unique_clusters);
  galois::GAccumulator<uint64_t> num_edges_acc;
  galois::do_all(galois::iterate((uint32_t)0, num_nodes_next),
                [&](uint32_t c){
                  prefix_edges_count[c] = edges_id[c].size();
                  num_edges_acc += prefix_edges_count[c];
                }, galois::steal());

  num_edges_next = num_edges_acc.reduce();
  for(uint32_t c = 1; c < num_nodes_next; ++c) {
    prefix_edges_count[c] += prefix_edges_count[c - 1];
  }

  assert(prefix_edges_count[num_unique_clusters - 1] == num_edges_next);
  galois::gPrint("#nodes : ", num_nodes_next, ", #edges : ", num_edges_next, "\n");
  galois::gPrint("#prefix last : ", prefix_edges_count[num_unique_clusters - 1], "\n");
#if 0
  for(uint32_t i = 0; i < num_nodes_next; ++i){
    for(uint32_t j = 0; j < edges_id[i].size(); ++j){
      galois::gPrint(i, " -B-> <", edges_id[i][j], " , ", edges_data[i][j], ">\n");
    }
  }
#endif

  std::cerr << "Graph construction started" << "\n";

  galois::StatTimer TimerConstructFrom("Timer_Construct_From");
  TimerConstructFrom.start();
  graph_next.constructFrom(num_nodes_next, num_edges_next, prefix_edges_count, edges_id, edges_data);
  TimerConstructFrom.stop();

	//setting flatSizes
	galois::do_all(galois::iterate(graph_next),
		[&] (GNode n){
			
			graph_next.getData(n).flatSize = cluster_flatsize[n];
			graph_next.getData(n).curr_comm_ass = comm_ass[n];
			graph_next.getData(n).curr_subcomm_ass = comm_ass[n];
		}, galois::steal());


  TimerGraphBuild.stop();
  galois::gPrint("Graph construction done\n");
}

void leiden(Graph &graph, largeArray& clusters_orig){


	Graph* graph_curr;
	Graph graph_next;

	graph_curr = &graph;	
	uint64_t num_nodes_orig = clusters_orig.size();

	double prev_quality = -INF_VAL;
	double curr_quality = -INF_VAL;

	int64_t iter = 0;

	galois::do_all(galois::iterate(*graph_curr),
      [&] (GNode n){

				graph.getData(n).curr_comm_ass = n;
		}, galois::steal());

	while(true){

		iter++;
		
		CommArray c_info;
		c_info.allocateBlocked(2*graph_curr->size()+1);

		sumVertexDegreeWeight(*graph_curr, c_info);
		moveNodesFast(*graph_curr, c_info);

		bool done = true;

		//check if done or not
		galois::do_all(galois::iterate(*graph_curr),
			[&] (GNode n){

				if(c_info[n].size > 1 && done){
					done = false;
				}
			}, galois::steal());

		//termination criterion
		if(done){
			galois::do_all(galois::iterate(*graph_curr),
				[&] (GNode n){
					graph_curr->getData(n).curr_subcomm_ass = graph_curr->getData(n).curr_comm_ass;
				}, galois::steal());
		uint64_t num_unique_clusters = renumberClustersContiguously(*graph_curr);
			galois::do_all(galois::iterate((uint64_t)0, num_nodes_orig),
                    [&](GNode n) {
                      if(clusters_orig[n] >= 0){
                        assert(clusters_orig[n] < (*graph_curr).size());
                        clusters_orig[n] = (*graph_curr).getData(clusters_orig[n], flag_no_lock).curr_subcomm_ass;
                     }
                    }, galois::steal());
	
	

			break;
		}

		refinePartition(*graph_curr);

		uint64_t num_unique_clusters = renumberClustersContiguously(*graph_curr);
		
			galois::do_all(galois::iterate((uint64_t)0, num_nodes_orig),
                    [&](GNode n) {
                    if(clusters_orig[n] >= 0){
                        assert(clusters_orig[n] < (*graph_curr).size());
                        clusters_orig[n] = (*graph_curr).getData(clusters_orig[n]).curr_subcomm_ass;
                      }
                    }, galois::steal());
	
		    prev_quality = curr_quality;
		
		//std::cout << "num unique:" << num_unique_clusters << std::endl;
		buildNextLevelGraph(*graph_curr, graph_next, num_unique_clusters);

		graph_curr = &graph_next;					

/*		switch(quality){
			case CPM:
				curr_quality = calCPMQualityFinal(*graph_curr);
				break;
			case Mod:
				curr_quality = calModularityFinal(*graph_curr);
				break;
			default:
				std::abort();
		}

		std::cout <<"Prev Quality: " << prev_quality << std::endl;
    std::cout << "Curr Quality: " << curr_quality << std::endl;
*/
		//destroying c_info
		c_info.destroy();
		c_info.deallocate();	
	
	}
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph, graph_next;
  Graph *graph_curr;

  galois::StatTimer TEnd2End("Timer_end2end");
  TEnd2End.start();

  std::cout << "Reading from file: " << filename << std::endl;
  std::cout << "[WARNING:] Make sure " << filename << " is symmetric graph without duplicate edges" << std::endl;
  galois::graphs::readGraph(graph, filename);
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges" << std::endl;

  graph_curr = &graph;

	setConstant(*graph_curr);
	setResolution(1.0f);
	setRandomness(0.01f);
  /*
   * To keep track of communities for nodes in the original graph.
   *Community will be set to -1 for isolated nodes
   */
  largeArray clusters_orig;
  clusters_orig.allocateBlocked(graph_curr->size());

	//initialize flat sizes to 1
	galois::do_all(galois::iterate(*graph_curr),
                  [&](GNode n){	

										graph.getData(n).flatSize = (uint64_t) 1;									
									}, galois::steal());

    /*
     *Initialize node cluster id.
     */
  galois::do_all(galois::iterate(*graph_curr),
                  [&](GNode n){
                    clusters_orig[n] = n;
                  }, galois::steal());

  printGraphCharateristics(*graph_curr);

  galois::gPrint("GOING in \n");
  galois::StatTimer Tmain("Timer_Leiden");
  Tmain.start();
  leiden(*graph_curr, clusters_orig);
	Tmain.stop();

  TEnd2End.stop();

  /*
   * Sanity check: Check modularity at the end
   */
//std::cout <<"res:" << resolution <<std::endl;
	switch(quality){
          case CPM:
						checkCPMQuality(graph, clusters_orig);
						break;
					case Mod:	
  					checkModularity(graph, clusters_orig);
						break;
					default:
						std::abort();
	}
  if(output_CID){
    printNodeClusterId(graph, output_CID_filename);
  }
  return 0;
}
