/**
 * This page rank application simply using MPI and nothing from Galois.
 *
 */


#include <mpi.h>
#include <iostream>
#include "galois/Galois.h"
#include "galois/Graph/FileGraph.h"

#include <list>
#include <bitset>
#include <string>
#include <new>

using namespace std;

//! d is the damping factor. Alpha is the prob that user will do a random jump, i.e., 1 - d
static const double alpha = (1.0 - 0.85);

//! maximum relative change until we deem convergence
static const double TOLERANCE = 0.1;

enum {
  MASTER = 0,
  DELTA,
  TIME,
  NUM_MSG_TYPE
};

int global_chunk_size;
struct Node {
  float rank;
  float residual;
  //std::atomic<float> residual;
  std::list<unsigned> out_edges;

};

size_t ReadFile(std::string inputFile, std::vector<Node>& graph_nodes) {
  galois::graphs::FileGraph fg;
  fg.fromFile(inputFile);

  size_t total_nodes = fg.size();

  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int host_ID;
  MPI_Comm_rank(MPI_COMM_WORLD, &host_ID);

  if(host_ID == 0) {
    global_chunk_size = ceil((float(total_nodes)/(size)));
    for (int dest = 1; dest < size; ++dest)
    MPI_Send(&global_chunk_size, 1, MPI_INT, dest, MASTER, MPI_COMM_WORLD);
  }
  else {
    MPI_Recv(&global_chunk_size, 1, MPI_INT, 0, MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  cout << " Chunk Size : " << global_chunk_size << "\n";
  int sent_data = global_chunk_size;


  //Let's send vectors around
  //std::vector<Node> graph_nodes;
  auto ii_fg = fg.begin();
  ii_fg += host_ID*global_chunk_size;

  if (std::distance(ii_fg, fg.end()) > global_chunk_size)
    graph_nodes.resize(global_chunk_size);
  else
    graph_nodes.resize(std::distance(ii_fg, fg.end()));

  for(int i = 0; (i < global_chunk_size) && (ii_fg != fg.end()); ++ii_fg, ++i) {
    Node n;
    n.rank = 1;
    n.residual = 1 - alpha;
    for (auto ii = fg.edge_begin(*ii_fg); ii != fg.edge_end(*ii_fg); ++ii) {
      unsigned dest = fg.getEdgeDst(ii);
      n.out_edges.push_back(dest);
    }
    graph_nodes[i] = n;
  }

 cout << " DeAllocating Filegraph : "<< host_ID << "\n";
 return total_nodes;
}


void PageRank_Bulk_AlltoAll(std::vector<Node>& graph_nodes, int chunk_size, int iterations, size_t total_nodes, std::vector<double> &time_compute, std::vector<double> &time_mpi, std::vector<double> &time_apply) {
//void PageRank_Bulk_AlltoAll(std::vector<Node>& graph_nodes, int chunk_size, int iterations, size_t total_nodes, std::vector<vector<double>> &time_compute, std::vector<vector<double>> &time_mpi, std::vector<vector<double>> &time_apply) {
//void PageRank_Bulk_AlltoAll(std::vector<Node>& graph_nodes, int chunk_size, int iterations, size_t total_nodes) {

  int num_hosts;
  MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);

  int host_ID;
  MPI_Comm_rank(MPI_COMM_WORLD, &host_ID);

  std::cout << "[ " << host_ID << " ] Inside PR\n";

  int local_chunk_size = graph_nodes.size();
  vector<float>send_buffer(total_nodes, 0.0);
  vector<float>receive_buffer(total_nodes, 0.0);

  double start_time_compute, end_time_compute, start_time_apply, end_time_apply, start_time_mpi, end_time_mpi;
  vector<double> send_times(3, 0.0);

  for(int iter = 0; iter < iterations; ++iter) {

    start_time_compute = MPI_Wtime();
    for(auto& GNode : graph_nodes) {
      int neighbors = GNode.out_edges.size();
      float old_residual = GNode.residual;
      GNode.residual = 0.0; // TODO: does old residual goes to zero
      GNode.rank += old_residual;
      float delta = old_residual*alpha/neighbors;

      if (delta > 0) {
        // Bulk Syncronous for all the nodes
        for (auto& nb : GNode.out_edges) {
            send_buffer[nb] += delta;
        }
      }
    }

    end_time_compute = MPI_Wtime();

    int status;
    start_time_mpi = MPI_Wtime();

    status = MPI_Alltoall(&send_buffer.front(), chunk_size, MPI_FLOAT, &receive_buffer.front(), chunk_size, MPI_FLOAT, MPI_COMM_WORLD);
    if(status != MPI_SUCCESS) {
      cout << "MPI_Alltoall failed\n";
      exit(0);
    }

    end_time_mpi = MPI_Wtime();

    start_time_apply = MPI_Wtime();

    // Apply received delta.
    int index = 0;
    for(int j = 0; j < total_nodes; ++j) {
      graph_nodes[index].residual += receive_buffer[j];
      index++;
      if(index >= local_chunk_size)
        index = 0;
    }
    end_time_apply = MPI_Wtime();


    //DEBUG: For computing individual times.

    send_times[0] += (end_time_compute - start_time_compute);
    send_times[1] += (end_time_mpi - start_time_mpi);
    send_times[2] += (end_time_apply - start_time_apply);
/*
 *  double send_times[num_hosts*3];
    send_times[3*host_ID + 0] = (end_time_compute - start_time_compute);
    send_times[3*host_ID + 1] = (end_time_mpi - start_time_mpi);
    send_times[3*host_ID + 2] = (end_time_apply - start_time_apply);

    double recv_times[3*num_hosts];
    status = MPI_Alltoall(send_times, 3, MPI_DOUBLE, recv_times, 3, MPI_DOUBLE, MPI_COMM_WORLD);

    time_compute[host_ID]  += recv_times[3*host_ID + 0];
    time_mpi[host_ID] += recv_times[3*num_hosts + 1];
    time_apply[host_ID] += recv_times[3*num_hosts + 2];
*/

  }


  // DEBUG: Send times to zero
  if(host_ID != 0)
  {
    MPI_Send(&send_times.front(), 3, MPI_DOUBLE, 0, TIME, MPI_COMM_WORLD);
  }
  if(host_ID == 0)
  {
    time_compute[host_ID]  = send_times[0];
    time_mpi[host_ID]      = send_times[1];
    time_apply[host_ID]    = send_times[2];

    vector<double> recv_times(3, 0.0);
    MPI_Status recv_status;
    for(int src = 1; src < num_hosts; src++)
    {
      MPI_Recv(&recv_times.front(), 3, MPI_DOUBLE, src, TIME, MPI_COMM_WORLD, &recv_status);

      time_compute[src] = recv_times[0];
      time_mpi[src]     = recv_times[1];
      time_apply[src]   = recv_times[2];
    }
  }

}

int main(int argc, char** argv){

  MPI_Init(NULL, NULL);


  global_chunk_size = 0;
  int hosts;
  MPI_Comm_size(MPI_COMM_WORLD, &hosts);

  int host_ID;
  MPI_Comm_rank(MPI_COMM_WORLD, &host_ID);

  cout << " DMPI_Init : "<< host_ID << "\n";

  double start_time, end_time;

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);


  // Reading inputFile in host 0
  if (argc < 2) {
    cout << "Usage : provide inputFile name\n";
    exit(0);
  }

  std::string inputFile = argv[1];
  //cout << "File : " << inputFile << "\n";

  int iterations = 2;
  if (argc == 3) {
    iterations = atoi(argv[2]);
  }

  // Timing vectors
  //std::vector<vector<double>> time_compute(hosts, vector<double>(iterations, 0.0));
  //std::vector<vector<double>> time_mpi(hosts, vector<double>(iterations, 0.0));
  //std::vector<vector<double>> time_apply(hosts, vector<double>(iterations, 0.0));

  std::vector<double> time_compute(hosts,0.0);
  std::vector<double> time_mpi(hosts,0.0);
  std::vector<double> time_apply(hosts,0.0);


  if(host_ID == 0) {
    cout << "\n All should be zero \n";
    cout << "\n --------------- Time Table --------------------------\n";
    for(int i = 0 ; i < hosts; ++i)
    {
      cout << "Host : " << i << "\n";
      cout << "\n\t " << time_compute[i] << "   " <<time_mpi[i] << "  "<< time_apply[i] << "\n";
    }
  }




  // Per host Graph
  std::vector<Node> graph_nodes;

  cout << " Calling ReadFile : "<< host_ID << "\n";
  size_t total_nodes = ReadFile(inputFile, graph_nodes);

  cout << " Host : " << host_ID << " Graph nodes : " << graph_nodes.size() << "\n";
  cout << " Host : " << host_ID << " Total nodes : " << total_nodes << "\n";

  cout << "Calling Bulk_AlltoAll : "<< host_ID << "\n";

  /* Marks the beginning of MPI process */
  MPI_Barrier(MPI_COMM_WORLD);
  start_time = MPI_Wtime();

  PageRank_Bulk_AlltoAll(graph_nodes, global_chunk_size, iterations, total_nodes, time_compute, time_mpi, time_apply);

  MPI_Barrier(MPI_COMM_WORLD);
  end_time = MPI_Wtime();


  MPI_Finalize();

  if(host_ID == 0) {
    cout << "\n Accumulative time for " << iterations << " iterations\n";
    cout << "\n --------------- Time Table --------------------------\n";
    for(int i = 0 ; i < hosts; ++i)
    {
      cout << "Host : " << i << "\n";
      cout << "\n\t " << time_compute[i] << "   " << time_mpi[i] << "  "<< time_apply[i] << "\n";
      //cout << "\n\t " << std::accumulate(time_compute[i].begin(), time_compute[i].end(), 0.0) << "   " << std::accumulate(time_mpi[i].begin(), time_mpi[i].end(), 0.0) << "  "<< std::accumulate(time_apply[i].begin(), time_apply[i].end(), 0.0) << "\n";
    }
    //cout << "\n\n " << time_compute[0] << "   " << time_mpi[0] << "  "<< time_apply[0] << "\n";
    //cout << "\n Accumulative time for " << iterations << " iterations\n";
    //cout << "\n\n " << std::accumulate(time_compute.begin(), time_compute.end(), 0.0) << "   " << std::accumulate(time_mpi.begin(), time_mpi.end(), 0.0) << "  "<< std::accumulate(time_apply.begin(), time_apply.end(), 0.0) << "\n";
    cout << "\n\n RUNTIME : " << end_time - start_time << " Seconds \n";
  }

  return 0;
}

