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


#ifndef LOUVAIN_CLUSTERING_H
#define LOUVAIN_CLUSTERING_H

/*
 * Typedefs
 */

constexpr static const uint64_t INF_VAL = std::numeric_limits<uint64_t>::max() / 2 - 1;

constexpr galois::MethodFlag flag_no_lock = galois::MethodFlag::UNPROTECTED;
constexpr galois::MethodFlag flag_read_lock = galois::MethodFlag::READ;
constexpr galois::MethodFlag flag_write_lock = galois::MethodFlag::WRITE;

typedef galois::LargeArray<int64_t> largeArray;

//Maintain community information
struct Comm {
  std::atomic<uint64_t> size;
  std::atomic<uint64_t> degree_wt;
	std::atomic<uint64_t> flatSize;
	std::atomic<uint64_t> external_edge_wt;
};

typedef galois::LargeArray<Comm> CommArray;

//Graph Node information
struct Node{
  int64_t prev_comm_ass;
  int64_t curr_comm_ass;
	
	int64_t prev_subcomm_ass;
	int64_t curr_subcomm_ass;

  uint64_t degree_wt;
  //uint64_t cluster_wt_internal;
  int64_t colorId;

	uint64_t flatSize;
	uint64_t external_edge_wt;

	bool inBag;
};

typedef uint64_t EdgeTy;
using Graph =
    galois::graphs::LC_CSR_Graph<Node, EdgeTy>::with_no_lockable<false>::type::with_numa_alloc<true>::type;

using GNode = Graph::GraphNode;

igraph_rng_t rng;
double resolution;

void setResolution(double res){

	resolution = res;
}


void printGraphCharateristics(Graph& graph) {

  galois::gPrint("/******************************************/\n");
  galois::gPrint("/************ Graph Properties ************/\n");
  galois::gPrint("/******************************************/\n");
  galois::gPrint("Number of Nodes: ", graph.size(), "\n");
  galois::gPrint("Number of Edges: ", graph.sizeEdges(), "\n");

}

uint64_t vertexFollowing(Graph& graph){

  //Initialize each node to its own cluster
  galois::do_all(galois::iterate(graph),
                 [&graph](GNode n) { graph.getData(n).curr_comm_ass = n; });

  //Remove isolated and degree-one nodes
  galois::GAccumulator<uint64_t> isolatedNodes;
  galois::do_all(galois::iterate(graph),
                  [&](GNode n) {
                    auto& n_data = graph.getData(n);
                    uint64_t degree = std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                                                     graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
                    if(degree == 0) {
                      isolatedNodes += 1;
                      n_data.curr_comm_ass = -1;
                    } else {
                      if(degree == 1) {
                        //Check if the destination has degree greater than one
                        auto dst = graph.getEdgeDst(graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
                        uint64_t dst_degree = std::distance(graph.edge_begin(dst, galois::MethodFlag::UNPROTECTED),
                                                         graph.edge_end(dst, galois::MethodFlag::UNPROTECTED));
                        if((dst_degree > 1 || (n > dst))){
                          isolatedNodes += 1;
                          n_data.curr_comm_ass = graph.getData(dst).curr_comm_ass;
                        }
                      }
                    }
                  });
  //The number of isolated nodes that can be removed
  return isolatedNodes.reduce();
}



void sumVertexDegreeWeight(Graph& graph, CommArray& c_info) {
//void sumVertexDegreeWeight(Graph& graph, std::vector<Comm>& c_info) {
  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  uint64_t total_weight = 0;
                  auto &n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii){
                     total_weight += graph.getEdgeData(ii, flag_no_lock);
                  }
                  n_data.degree_wt = total_weight;
                  c_info[n].degree_wt = total_weight;
                  //galois::gPrint(n, " : ", c_info[n].degree_wt.load(), "\n");
                  c_info[n].size = 1;
                });
}

void sumClusterWeight(Graph& graph, CommArray& c_info) {
  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  uint64_t total_weight = 0;
                  auto &n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii){
                     total_weight += graph.getEdgeData(ii, flag_no_lock);
                  }
                  n_data.degree_wt = total_weight;
                  c_info[n].degree_wt = 0;
                });

  /*
   * TODO: Parallelize this
   */
  for(GNode n = 0; n < graph.size(); ++n) {
      auto &n_data = graph.getData(n);
      if(n_data.curr_comm_ass >= 0)
        c_info[n_data.curr_comm_ass].degree_wt += n_data.degree_wt;
    }
}

double calConstantForSecondTerm(Graph& graph){
  galois::GAccumulator<uint64_t> local_weight;
  galois::do_all(galois::iterate(graph),
                [&graph, &local_weight](GNode n){
                  local_weight += graph.getData(n).degree_wt;
                });
  /* This is twice since graph is symmetric */
  uint64_t total_edge_weight_twice = local_weight.reduce();
  return 1/(double)total_edge_weight_twice;
}


uint64_t maxModularity(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc, double constant) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
  double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  double ay = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y
      eiy = counter[stored_already->second]; // Total edges incident on cluster y
      //cur_gain = 2 * (eiy - eix) - 2 * degree_wt * (ay - ax) * constant;
      //From the paper: Verbatim
      cur_gain = 2 * constant * (eiy - eix) + 2 * degree_wt * ((ax - ay) * constant * constant);

      if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && (stored_already->first < max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  //galois::gPrint("Max Gain : ", max_gain, "\n");
  //if(max_gain < 1e-3 || (c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
  if((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
    max_index = sc;
  }

  assert(max_gain >= 0);
  return max_index;
}

uint64_t maxModularityWithoutSwaps(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc, double constant) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
  double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  double ay = 0;

	double size_x = c_info[sc].size;
	double size_y = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y

			size_y = c_info[stored_already->first].size;
			//if(ay < (ax + degree_wt)){
			if(size_y < size_x){
				stored_already++;	
				continue;
			}
			else if (size_y == size_x && stored_already->first > sc)
			{
				stored_already++;
				continue;
			}

      eiy = counter[stored_already->second]; // Total edges incident on cluster y
      //cur_gain = 2 * (eiy - eix) - 2 * degree_wt * (ay - ax) * constant;
      //From the paper: Verbatim
      cur_gain = 2 * constant * (eiy - eix) + 2 * degree_wt * ((ax - ay) * constant * constant);

      if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && (stored_already->first < max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  //galois::gPrint("Max Gain : ", max_gain, "\n");
  //if(max_gain < 1e-3 || (c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
  if((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
    max_index = sc;
  }

  assert(max_gain >= 0);
  return max_index;
}


double calModularityDelay(Graph& graph, CommArray& c_info, CommArray& c_update, double& e_xx, double& a2_x, double& constant_for_second_term, std::vector<GNode>& local_target) {

  /* Variables needed for Modularity calculation */
  double mod = -1;

  largeArray cluster_wt_internal;


  /*** Initialization ***/
  cluster_wt_internal.allocateBlocked(graph.size());


   /* Calculate the overall modularity */
  //double e_xx = 0;
  galois::GAccumulator<double> acc_e_xx;
  //double a2_x = 0;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(local_target[graph.getEdgeDst(ii)] == local_target[n]) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].degree_wt + c_update[n].degree_wt) * ((double) (c_info[n].degree_wt + c_update[n].degree_wt) * (double)constant_for_second_term);
                });


  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
  mod = e_xx * (double)constant_for_second_term - a2_x * (double)constant_for_second_term;
  //galois::gPrint("Final Stats: ", " Number of clusters:  ", graph.size() , " Modularity: ", mod, "\n");

  return mod;
}



double calModularity(Graph& graph, CommArray& c_info, double& e_xx, double& a2_x, double& constant_for_second_term) {

  /* Variables needed for Modularity calculation */
  double mod = -1;

  largeArray cluster_wt_internal;


  /*** Initialization ***/
  cluster_wt_internal.allocateBlocked(graph.size());


   /* Calculate the overall modularity */
  //double e_xx = 0;
  galois::GAccumulator<double> acc_e_xx;
  //double a2_x = 0;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  auto n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(graph.getData(graph.getEdgeDst(ii)).curr_comm_ass == n_data.curr_comm_ass) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].degree_wt) * ((double) (c_info[n].degree_wt) * (double)constant_for_second_term);
                });


  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
  mod = e_xx * (double)constant_for_second_term - a2_x * (double)constant_for_second_term;
  //galois::gPrint("Final Stats: ", " Number of clusters:  ", graph.size() , " Modularity: ", mod, "\n");

  return mod;
}

double calModularityFinal(Graph& graph) {
  CommArray c_info; // Community info
  CommArray c_update; // Used for updating community

  /* Variables needed for Modularity calculation */
  double constant_for_second_term;
  double mod = -1;

  largeArray cluster_wt_internal;
  //std::vector<uint64_t> cluster_wt_internal(graph.size());


  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());
  cluster_wt_internal.allocateBlocked(graph.size());

  /* Calculate the weighted degree sum for each vertex */
  sumClusterWeight(graph, c_info);

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);


   /* Calculate the overall modularity */
  double e_xx = 0;




  galois::GAccumulator<double> acc_e_xx;
  double a2_x = 0;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  auto n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(graph.getData(graph.getEdgeDst(ii)).curr_comm_ass == n_data.curr_comm_ass) {
                    //if(graph.getData(graph.getEdgeDst(ii)).prev_comm_ass == n_data.prev_comm_ass) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].degree_wt) * ((double) (c_info[n].degree_wt)* (double)constant_for_second_term);
								});

	
	e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();
	

	mod = e_xx * (double)constant_for_second_term - a2_x * (double)constant_for_second_term;
  return mod;
}

void set_rng(){
	
	igraph_rng_init(&rng, &igraph_rngtype_mt19937);
	igraph_rng_seed(&rng, rand());
}

uint64_t getRandomInt(uint64_t from, uint64_t to){
	
	return igraph_rng_get_integer(rng, from, to);
}

double diffCPMQuality(uint64_t curr_subcomm, uint64_t candidate_subcomm, std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, CommArray &subcomm_info, uint64_t self_loop_wt){

	uint64_t size_x = subcomm_info[curr_subcomm].flatSize;
	uint64_t size_y = subcomm_info[candidate_subcomm].flatSize;

	double diff = (double)(counter[cluster_local_map[candidate_subcomm]] - counter[cluster_local_map[curr_subcomm]] + self_loop_wt) + resolution * 0.5f*(double)((size_x*(size_x-1) + size_y*(size_y-1)) - ((size_x-1)*(size_x-2) + size_y*(size_y+1)));

	return diff;
}
//subcomm_info should have updated size values
uint64_t getRandomSubcommunity(Graph& graph, uint64_t n, CommArray &subcomm_info, uint64_t flatSize_comm){

	uint64_t rand_subcomm = -1;
	uint64_t curr_subcomm = graph.getData(n).curr_subcomm_ass;

	std::map<uint64_t, uint64_t> cluster_local_map; // Map each neighbor's subcommunity to local number: Subcommunity --> Index
	std::vector<uint64_t> counter; //Number of edges to each unique subcommunity
	uint64_t num_unique_clusters = 1;

	cluster_local_map[curr_subcomm] = 0; // Add n's current subcommunity
	counter.push_back(0); //Initialize the counter to zero (no edges incident yet)

	uint64_t self_loop_wt = 0;

	for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
  	GNode dst = graph.getEdgeDst(ii);
    auto edge_wt = graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded	

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
	std::vector<uint64_t> new_counter;		
	num_unique_clusters = 0;
	uint64_t total = 0;	

	for(auto pair: cluster_local_map){

		auto subcomm = pair.first;
		if(curr_subcomm == subcomm)
			continue;

		double flatSize_subcomm = (double) subcomm_info[subcomm].flatSize;

		//check if subcommunity is well connected
		if(subcomm_info[subcomm].external_edge_wt < resolution*flatSize_subcomm*((double)flatSize_comm - flatSize_subcomm))
			continue;

		if(diffCPMQuality(curr_subcomm, subcomm, cluster_local_map, counter, subcomm_info, self_loop_wt) > 0){
			new_cluster_local_map[subcomm] = num_unique_clusters;
			uint64_t count = counter[cluster_local_map[subcomm]];
			new_counter.push_back(count);
			total += count;
		}
	}

	uint64_t rand_idx = getRandomInt(0,total-1);

	uint64_t idx = 0;
	for(auto pair: new_cluster_local_map){

		if(new_counter[idx] > rand_idx)
			return pair.first;

		rand_idx = rand_idx - new_counter[idx];
	}

	return -1;
}
/*
uint64_t maxCPMQuality(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc, double resolution) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
  double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  double ay = 0;

	double size_x = c_info[sc].size;

  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y
			size_y = c_info[stored_already->first].size;

      eiy = counter[stored_already->second]; // Total edges incident on cluster y
      //cur_gain = 2 * (eiy - eix) - 2 * degree_wt * (ay - ax) * constant;
      //From the paper: Verbatim
      cur_gain = 2 * (eiy - eix) + resolution*(size_x*(size_x - 1) - (size_y+1)*size_y);

      if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && (stored_already->first > max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
*/

uint64_t maxCPMQuality(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc, double resolution) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
  double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  double ay = 0;

	double size_x = c_info[sc].flatSize;

  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y
			size_y = c_info[stored_already->first].flatSize;

      eiy = counter[stored_already->second]; // Total edges incident on cluster y
      //cur_gain = 2 * (eiy - eix) - 2 * degree_wt * (ay - ax) * constant;
      //From the paper: Verbatim
   		cur_gain = 2.0f * (double)(eiy - eix) + resolution*((double)(size_x*(size_x - 1) + size_y*(size_y - 1)) - (double)((size_x-1)*(size_x-2) + (size_y+1)*size_y));   

      if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && (stored_already->first > max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  //galois::gPrint("Max Gain : ", max_gain, "\n");
  //if(max_gain < 1e-3 || (c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
 // if((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
   // max_index = sc;
  //}

  assert(max_gain >= 0);
  return max_index;
}

	

uint64_t maxCPMQualityWithoutSwaps(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc) {

  int64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
  double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  double ay = 0;

	double size_x = c_info[sc].flatSize;
	double size_y = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y

			size_y = c_info[stored_already->first].flatSize;
			//if(ay < (ax + degree_wt)){
			if(size_y < size_x){
				stored_already++;	
				continue;
			}
			else if (size_y == size_x && stored_already->first < sc)
			{
				stored_already++;
				continue;
			}

      eiy = counter[stored_already->second]; // Total edges incident on cluster y
      //cur_gain = 2 * (eiy - eix) - 2 * degree_wt * (ay - ax) * constant;
      //From the paper: Verbatim
      cur_gain = 2.0f * (double)(eiy - eix) + resolution*((double)(size_x*(size_x - 1) + size_y*(size_y - 1)) - (double)((size_x-1)*(size_x-2) + (size_y+1)*size_y));

      if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && (stored_already->first > max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  //galois::gPrint("Max Gain : ", max_gain, "\n");
  //if(max_gain < 1e-3 || (c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
  //if((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
   // max_index = sc;
 // }

  assert(max_gain >= 0);
  return max_index;
}


//c_info must have updated size values
double calCPMQuality(Graph& graph, CommArray& c_info, double& e_xx, double& a2_x, double resolution) {

  /* Variables needed for CPM quality calculation */
  double mod = -1;

  largeArray cluster_wt_internal;


  /*** Initialization ***/
  cluster_wt_internal.allocateBlocked(graph.size());


   /* Calculate the overall modularity */
  //double e_xx = 0;
  galois::GAccumulator<double> acc_e_xx;
  //double a2_x = 0;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  auto n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(graph.getData(graph.getEdgeDst(ii)).curr_comm_ass == n_data.curr_comm_ass) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].flatSize) * ((double) (c_info[n].flatSize - 1) * 0.5f;
                });


  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
  mod = e_xx - a2_x*resolution;
  //galois::gPrint("Final Stats: ", " Number of clusters:  ", graph.size() , " Modularity: ", mod, "\n");

  return mod;
}



/*
 * To compute the final modularity using prev cluster
 * assignments.
 */
double calCPMQualityFinal(Graph& graph, double resolution) {
  CommArray c_info; // Community info

  /* Variables needed for Modularity calculation */
  double mod = -1;

  largeArray cluster_wt_internal;
  //std::vector<uint64_t> cluster_wt_internal(graph.size());


  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  cluster_wt_internal.allocateBlocked(graph.size());


   /* Calculate the overall modularity */
  double e_xx = 0;
  galois::GAccumulator<double> acc_e_xx;
  double a2_x = 0;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  auto n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(graph.getData(graph.getEdgeDst(ii)).curr_comm_ass == n_data.curr_comm_ass) {
                    //if(graph.getData(graph.getEdgeDst(ii)).prev_comm_ass == n_data.prev_comm_ass) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }

									//updating community size
									c_info[n_data.curr_comm_ass].size += (uint64_t)1;
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].flatSize) * ((double) (c_info[n].flatSize - 1)* 0.5f);
                });


  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
  mod = e_xx - a2_x*resolution;
  return mod;
}




//need to update this
uint64_t renumberClustersContiguously(Graph &graph) {

  std::map<uint64_t, uint64_t> cluster_local_map;
  uint64_t num_unique_clusters = 0;

  for (GNode n = 0; n < graph.size(); ++n){
    auto& n_data = graph.getData(n, flag_no_lock);
    if(n_data.curr_comm_ass != -1) {
      assert(n_data.curr_comm_ass < graph.size());
      auto stored_already = cluster_local_map.find(n_data.curr_comm_ass);
     if(stored_already != cluster_local_map.end()){
      n_data.curr_comm_ass = stored_already->second;
     } else {
      cluster_local_map[n_data.curr_comm_ass] = num_unique_clusters;
      n_data.curr_comm_ass = num_unique_clusters;
      num_unique_clusters++;
     }
    }
  }

  return num_unique_clusters;
}

uint64_t renumberClustersContiguouslyArray(largeArray &arr) {

  std::map<uint64_t, uint64_t> cluster_local_map;
  uint64_t num_unique_clusters = 0;

  for (GNode n = 0; n < arr.size(); ++n){
    if(arr[n] != -1) {
      assert(arr[n] < arr.size());
      auto stored_already = cluster_local_map.find(arr[n]);
     if(stored_already != cluster_local_map.end()){
      arr[n] = stored_already->second;
     } else {
      cluster_local_map[arr[n]] = num_unique_clusters;
      arr[n] = num_unique_clusters;
      num_unique_clusters++;
     }
    }
  }



void printGraph(Graph& graph){
  for(GNode n = 0; n < graph.size(); ++n) {
    for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      galois::gPrint(n, " --> ", graph.getEdgeDst(ii), " , ", graph.getEdgeData(ii), "\n");
    }
  }
}

void printNodeClusterId(Graph& graph, std::string output_CID_filename){
  std::ofstream outputFile(output_CID_filename, std::ofstream::out);
  for(GNode n = 0; n < graph.size(); ++n) {
    outputFile << n << "  " << graph.getData(n).curr_comm_ass << "\n";
    //outputFile << graph.getData(n).curr_comm_ass << "\n";
  }
}

void checkModularity(Graph& graph, largeArray& clusters_orig) {
  galois::gPrint("checkModularity\n");

  //galois::gPrint("Number of unique clusters (renumber) ARR: ", renumberClustersContiguouslyArray(clusters_orig), "\n");
  galois::do_all(galois::iterate(graph),
                [&](GNode n){
                  graph.getData(n, flag_no_lock).curr_comm_ass = clusters_orig[n];
                });

  uint64_t num_unique_clusters = renumberClustersContiguously(graph);
  galois::gPrint("Number of unique clusters (renumber): ", num_unique_clusters, "\n");
  auto mod = calModularityFinal(graph);
  galois::gPrint("FINAL MOD: ", mod, "\n");
}

#endif //LOUVAIN_CLUSTERING_H
