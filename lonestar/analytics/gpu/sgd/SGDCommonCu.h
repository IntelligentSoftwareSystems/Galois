/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef SGDCOMMON_CU_H_
#define SGDCOMMON_CU_H_

#include <assert.h>
#define _SGD_USE_SHARED_MEM_ 1
#define SGD_FEATURE_SIZE 16

#ifdef _WIN32
template <typename T>
bool isnormal(const T&) {
  return true;
}
#else
using namespace std;

const float SGD_LAMBDA        = 0.05f;
const float SGD_LEARNING_RATE = 0.012f;
const float SGD_DECAY_RATE    = 0.015f;
const int SGD_MAX_ROUNDS      = 5;
typedef float FeatureType;

/************************************************
 *
 *************************************************/
float SGD_STEP_SIZE(int X) {
  return SGD_LEARNING_RATE * 1.5f / (1.0f + SGD_DECAY_RATE * pow(X + 1, 1.5f));
} // Purdue.
//#define SGD_STEP_SIZE(X) (0.001f *1.5f/(1.0+0.9* pow(X+1,1.5))) //Intel.
/************************************************
 *
 *************************************************/
float sum_vector(const float* a) {
  float res = 0.0f;
  for (int i = 0; i < SGD_FEATURE_SIZE; ++i)
    res += a[i];
  return res;
}
/************************************************
 *
 *************************************************/
template <typename T>
T dot_product(const T* a, const T* b) {
  T res = 0.0f;
  for (int i = 0; i < SGD_FEATURE_SIZE; i++) {
    assert((a[i] == 0 || isnormal(a[i]) == true));
    assert((b[i] == 0 || isnormal(b[i]) == true));
    res += a[i] * b[i];
  }
  return res;
}
/************************************************
 *
 *************************************************/
float toMB(long val) { return val / (float)(1024 * 1024); }
/************************************************
 *
 *************************************************/
struct DebugData {
  struct NodeStats {
    int max_rating;
    int min_rating;
    int sum_rating;
    int count_rating;
    int my_degree;
    bool is_movie;
    NodeStats() {
      max_rating = sum_rating = count_rating = 0;
      min_rating                             = std::numeric_limits<int>::max();
      my_degree                              = 0;
      is_movie                               = false;
    }
    void stat(int val) {
      max_rating = std::max(max_rating, val);
      min_rating = std::min(min_rating, val);
      sum_rating += (val);
      count_rating++;
    }
  };
  std::vector<std::pair<int, int>> user_degrees;
  std::vector<std::pair<int, int>> movie_degrees;
  std::map<int, int> user_map;
  std::map<int, int> movie_map;
};
/************************************************
 *
 *************************************************/

template <typename GraphTy>
static void write_stats_to_file(GraphTy& graph) {
  // 0 Write graph as csv to file:
  {
    std::ofstream out_file("/workspace/rashid/bgg.csv");
    out_file << "Src,Dst,Wt\n";
    for (size_t i = 0; i < graph.num_edges(); ++i) {
      out_file << graph.get_edge_src(i) << "," << graph.out_neighbors()[i]
               << "," << graph.out_edge_data()[i] << "\n";
    }
    out_file.close();
  }
  //      return;
  // 3 Write average-user degree per-movie
  {
    int max_user_degree  = 0;
    int max_movie_degree = 0;
    std::vector<DebugData::NodeStats> all_nodes_stats(graph.num_nodes());
    std::vector<int> movie_indices;
    std::vector<int> user_indices;
    {
      for (size_t i = 0; i < graph.num_edges(); ++i) {
        int src    = graph.get_edge_src(i);
        int dst    = graph.out_neighbors()[i];
        int rating = graph.out_edge_data()[i];
        all_nodes_stats[src].my_degree++;
        all_nodes_stats[dst].my_degree++;
        all_nodes_stats[src].is_movie = true;
        all_nodes_stats[src].stat(rating);
        all_nodes_stats[dst].stat(rating);
        movie_indices.push_back(src);
        user_indices.push_back(dst);
        max_movie_degree =
            std::max(max_movie_degree, all_nodes_stats[src].my_degree);
        max_user_degree =
            std::max(max_user_degree, all_nodes_stats[dst].my_degree);
      }
    }
    {
      /*std::sort(debug_data.user_degrees.begin(),
       debug_data.user_degrees.end(), [](const std::pair<int, int>& lhs, const
       std::pair<int, int>& rhs) { return lhs.second > rhs.second;});
       std::sort(debug_data.movie_degrees.begin(),
       debug_data.movie_degrees.end(), [](const std::pair<int, int>& lhs, const
       std::pair<int, int>& rhs) { return lhs.second > rhs.second;});
       */
    }
    std::vector<DebugData::NodeStats> user_per_degree_stats(max_user_degree +
                                                            1);
    std::vector<DebugData::NodeStats> movie_per_degree_stats(max_movie_degree +
                                                             1);
    long sum_ratings = 0;
    for (size_t i = 0; i < graph.num_edges(); ++i) {
      int m      = graph.get_edge_src(i);
      int u      = graph.out_neighbors()[i];
      int rating = graph.out_edge_data()[i];
      int m_d    = all_nodes_stats[m].my_degree;
      int u_d    = all_nodes_stats[u].my_degree;
      assert(all_nodes_stats[m].is_movie == true &&
             all_nodes_stats[u].is_movie == false);
      user_per_degree_stats.at(u_d).stat(rating);
      movie_per_degree_stats.at(m_d).stat(rating);
      sum_ratings += rating;
    }
    std::cout << "Sizes:: " << movie_indices.size() << ", "
              << user_indices.size() << "\n";
    std::cout << "Max-degree:: " << max_movie_degree << ", " << max_user_degree
              << "\n";
    std::cout << "Average rating:: " << sum_ratings / (float)(graph.num_edges())
              << "\n";
    {
      std::ofstream out_file_u("/workspace/rashid/user_stats.csv");
      out_file_u << "Id,Degree,NumNodes,Min,Max,Sum\n";
      for (size_t i = 0; i < user_indices.size(); ++i) {
        int index = user_indices[i];
        assert(all_nodes_stats[index].is_movie == false);
        out_file_u << i << "," << all_nodes_stats[index].count_rating << ","
                   << all_nodes_stats[index].min_rating << ","
                   << all_nodes_stats[index].max_rating << ","
                   << all_nodes_stats[index].sum_rating << "\n";
      }
      out_file_u.close();
    }
    {
      std::ofstream out_file_u("/workspace/rashid/movie_stats.csv");
      out_file_u << "Id,Degree,NumNodes,Min,Max,Sum\n";
      for (size_t i = 0; i < movie_indices.size(); ++i) {
        int index = movie_indices[i];
        assert(all_nodes_stats[index].is_movie == true);
        out_file_u << i << "," << all_nodes_stats[index].count_rating << ","
                   << all_nodes_stats[index].min_rating << ","
                   << all_nodes_stats[index].max_rating << ","
                   << all_nodes_stats[index].sum_rating << "\n";
      }
      out_file_u.close();
    }
    {
      std::ofstream out_file_u("/workspace/rashid/bgg_user_average_degree.csv");
      out_file_u << "Degree,NumNodes,Min,Max,Sum\n";
      for (int i = 0; i < max_user_degree; ++i) {
        if (user_per_degree_stats[i].count_rating > 0)
          out_file_u << i << "," << user_per_degree_stats[i].count_rating << ","
                     << user_per_degree_stats[i].min_rating << ","
                     << user_per_degree_stats[i].max_rating << ","
                     << user_per_degree_stats[i].sum_rating << "\n";
      }
      out_file_u.close();
    }
    {
      std::ofstream out_file_m(
          "/workspace/rashid/bgg_movie_average_degree.csv");
      out_file_m << "Degree,NumNodes,Min,Max,Sum\n";
      for (int i = 0; i < max_movie_degree; ++i) {
        if (movie_per_degree_stats[i].count_rating > 0)
          out_file_m << i << "," << movie_per_degree_stats[i].count_rating
                     << "," << movie_per_degree_stats[i].min_rating << ","
                     << movie_per_degree_stats[i].max_rating << ","
                     << movie_per_degree_stats[i].sum_rating << "\n";
      }
      out_file_m.close();
    }
  }
  // 4 Write average-movie degree per-user
  std::cout << "Done writing debug info...\n";
  exit(-1);
}
/************************************************
 *
 ************************************************/
template <typename GraphType, typename FeatureArrayType>
float compute_err(GraphType& graph, FeatureArrayType* features,
                  int max_rating) {
  int fail_count = 0;
  float sum      = 0;
  for (unsigned int i = 0; i < features->size(); ++i) {
    float f = features->host_ptr()[i];
    sum += f;
    if ((f != 0 && isnormal(f) == false)) {
      fail_count++;
    }
  }
  // fprintf(stderr, "Failed:: %6.6g,Sum, %6.6g ", fail_count / (float)
  // (features->size()), sum);
  float accumulated_error = 0.0f;
  float max_err           = 0.0f;
  for (unsigned int i = 0; i < graph.num_edges(); ++i) {
    int src      = graph.get_edge_src(i);
    int dst      = graph.out_neighbors()[i];
    float rating = graph.out_edge_data()[i] / (float)max_rating;
    // if(src <0 || src >= graph.num_nodes() || dst < 0 || dst >=
    // graph.num_nodes())
    //     fprintf(stderr, " error at src %d and dst %d\n", src, dst);
    float computed_rating =
        dot_product(&features->host_ptr()[src * SGD_FEATURE_SIZE],
                    &features->host_ptr()[dst * SGD_FEATURE_SIZE]);
    float err = (computed_rating - rating);
    max_err   = std::max((double)max_err, (double)fabs(err));
    accumulated_error += err * err;
  }
  accumulated_error /= (float)graph.num_edges();
  //   float rms = std::sqrt((float) accumulated_error);
  float rms = sqrt((float)accumulated_error);
  // fprintf(stderr, "Average_error %.3f , max_error %.3f, RMS %.5f \n",
  // accumulated_error, max_err, rms);
  printf("RMS %.5f\n", rms);
  return rms;
}
/************************************************************************
 *
 ************************************************************************/
template <typename GraphType, typename FeatureArrayType, typename LockType>
void initialize_features_random(GraphType& graph, FeatureArrayType* features,
                                LockType* locks, std::vector<int>& movies) {
  using namespace std;

  FeatureType top = 1.0 / sqrt(SGD_FEATURE_SIZE);
  //   uniform_real_distribution<FeatureType> dist(0, top);
  //   mt19937 gen;
  /*      std::uniform_real_distribution<FeatureType> dist(-1.0f, 1.0f);*/
  FeatureType feature_sum = 0.0f, min_feature = top, max_feature = -top;
  // For each node, initialize features to random, and lock to be unlocked.
  for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
    locks->host_ptr()[i] = -1;
    FeatureType* features_l =
        &(features
              ->host_ptr()[i *
                           SGD_FEATURE_SIZE]); // graph.node_data()[i].features;
    for (int j = 0; j < SGD_FEATURE_SIZE; ++j) {
      (features_l[j] = rand() / (float)std::numeric_limits<int>::max());
      feature_sum += (features_l[j] = features_l[j] * top);
      max_feature = std::max(features_l[j], max_feature);
      min_feature = std::min(features_l[j], min_feature);
      assert(isnormal(features_l[j]) || features_l[j] == 0);
    }
    if (graph.num_neighbors(i) > 0)
      movies.push_back(i);
  }
  // std::cout << "initial features:: " << feature_sum << " , [" << min_feature
  // << " , " << max_feature;
}
/************************************************************************
 *
 ************************************************************************/
template <typename GraphType, typename FeatureArrayType>
void initialize_features_random(GraphType& graph, FeatureArrayType* features,
                                std::vector<int>& movies) {
  using namespace std;

  FeatureType top         = 1.0 / sqrt(SGD_FEATURE_SIZE);
  FeatureType feature_sum = 0.0f, min_feature = top, max_feature = -top;
  // For each node, initialize features to random, and lock to be unlocked.
  for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
    FeatureType* features_l =
        &(features
              ->host_ptr()[i *
                           SGD_FEATURE_SIZE]); // graph.node_data()[i].features;
    for (int j = 0; j < SGD_FEATURE_SIZE; j++) {
      (features_l[j] = rand() / (float)std::numeric_limits<int>::max());
      feature_sum += (features_l[j] = features_l[j] * top);
      max_feature = std::max(features_l[j], max_feature);
      min_feature = std::min(features_l[j], min_feature);
      assert(isnormal(features_l[j]) || features_l[j] == 0);
    }
    if (graph.num_neighbors(i) > 0)
      movies.push_back(i);
  }
  // std::cout << "initial features:: " << feature_sum << " , [" << min_feature
  // << " , " << max_feature; std::cout << "initial features: feature_sum " <<
  // feature_sum << " min_feature " << min_feature << " max_feature " <<
  // max_feature << "\n";
}
/************************************************************************
 *
 ************************************************************************/
/************************************************
 *
 *************************************************/
template <typename GraphType>
void diagonal_graph(GraphType& g, int num_nodes) {
  g.init(2 * num_nodes, num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    g.outgoing_index()[i] = i;
    g.get_edge_src()[i]   = i;
    g.out_neighbors()[i]  = i + num_nodes;
    g.out_edge_data()[i]  = 3;
  }
  for (int i = num_nodes; i < 2 * num_nodes; ++i) {
    g.outgoing_index()[i] = num_nodes;
  }
  g.outgoing_index()[2 * num_nodes] = num_nodes;

} // End complete_bipartitie
/************************************************
 *
 *************************************************/
template <typename GraphType>
void complete_bipartitie(GraphType& g, int num_movies, int num_users) {
  g.init(num_movies + num_users, num_users * num_movies);
  int index = 0;
  for (int i = 0; i < num_movies; ++i) {
    g.outgoing_index()[i] = index;
    for (int j = 0; j < num_users; ++j) {
      g.get_edge_src()[index + j] = i;
      //         g.out_neighbors()[index + j] = num_movies + ((j + i) %
      //         num_users);
      g.out_neighbors()[index + j] = num_movies + j;
      g.out_edge_data()[index + j] = 3;
    }
    index += num_users;
  }
  for (int i = num_movies; i < num_movies + num_users; ++i) {
    g.outgoing_index()[i] = index;
  }
  g.outgoing_index()[num_movies + num_users] = index;

  if (false) {
    std::ofstream out_file("gen_graph.csv");
    for (int i = 0; i < num_movies; ++i) {
      for (size_t nbr_idx = g.outgoing_index()[i];
           nbr_idx < g.outgoing_index()[i + 1]; ++nbr_idx) {
        out_file << g.out_neighbors()[nbr_idx] << ",";
      }
      out_file << "\n";
    }
    out_file.close();
  }
} // End complete_bipartitie
/************************************************
 *
 *************************************************/
template <typename GraphType>
void compute_err(GraphType& graph) {
  int fail_count    = 0;
  float sum         = 0;
  float sum_ratings = 0;
  for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
    for (int idx = 0; idx < SGD_FEATURE_SIZE; ++idx) {
      float f = graph.node_data()[i].features[idx];
      sum += f;
      if ((f != 0 && isnormal(f) == false)) {
        fail_count++;
      }
    }
  }
  // fprintf(stderr, "Failed:: %6.6g,Sum, %6.6g ", fail_count / (float)
  // (graph.num_nodes()*SGD_FEATURE_SIZE), sum);
  float accumulated_error = 0.0f;
  float max_err           = 0.0f;
  typedef typename GraphType::NodeDataType NodeDataType;
  NodeDataType* features = graph.node_data();
  for (unsigned int i = 0; i < graph.num_edges(); ++i) {
    unsigned int src = graph.out_edge_src()[i];
    unsigned int dst = graph.out_neighbors()[i];
    float rating     = graph.out_edge_data()[i];
    sum_ratings += rating;
    float computed_rating =
        dot_product(features[src], features[dst], graph.num_nodes());
    float err = (computed_rating - rating);
    max_err   = std::max((double)max_err, (double)fabs(err));
    accumulated_error += err * err;
  }
  accumulated_error /= (float)graph.num_edges();
  float rms = sqrt((float)accumulated_error);
  // fprintf(stderr, "Average_error, %6.6f , max_error, %6.6f, RMS, %6.6f ,
  // RatingsSum, %6.6g\n", accumulated_error, max_err, rms,sum_ratings);
}
/************************************************
 *
 *************************************************/

#endif // OpenCL.
#endif /* SGDCOMMON_H_ */
