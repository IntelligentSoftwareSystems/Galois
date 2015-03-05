/** Page rank application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Gurbinder Gill <gill@cs.utexas.edu>
 */

/**
 * This page rank application simply using MPI and nothing from Galois.
 *
 */


#include <mpi.h>
#include <iostream>
#include "Galois/Galois.h"
#include "Galois/Graph/FileGraph.h"

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
  Galois::Graph::FileGraph fg;
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

  // destroy filegraph don't need it anymore.
  fg.~FileGraph();

  cout << " DeAllocating Filegraph : "<< host_ID << "\n";
 return total_nodes;
}


void receive() {

  int count = 0;
  MPI_Status status;

  int host_ID;
  MPI_Comm_rank(MPI_COMM_WORLD, &host_ID);

  MPI_Probe(0,0, MPI_COMM_WORLD, &status);
  MPI_Get_count(&status, MPI_INT, &count);

  std::vector<Node> recv_buffer(count);

  cout << "Host : " << host_ID << " : received chunk size : " << count << "\n";

//  MPI_Recv(&recv_buffer.front(), count, Node, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for(auto x : recv_buffer)
    cout << x.rank << "\t";
  cout << "\n";

}

int get_host(int node){

    return (node/global_chunk_size);
}


std::vector<float>::iterator find_if_index (std::vector<float> &vec, int i) {
  for (auto ii = vec.begin(); ii != vec.end(); ii+=2) {
    if(*ii == i )
      return ii;
  }
  return vec.end();
}

void PageRank_Bulk(std::vector<Node>& graph_nodes, int chunk_size, int iterations) {

  int num_hosts;
  MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);

  int host_ID;
  MPI_Comm_rank(MPI_COMM_WORLD, &host_ID);

  int send_count[num_hosts];
  int recv_count[num_hosts];

  std::vector<std::vector<float>> recv_buffer(num_hosts);
  std::vector<std::vector<float>> send_buffer(num_hosts);
  std::vector<std::vector<bool>> host_bitset(num_hosts, std::vector<bool>(chunk_size));

  for(int iter = 0; iter < iterations; ++iter) {
    cout << "Iteration : " << iter << " On host : " << host_ID << "\n";
    for(auto& GNode : graph_nodes) {

      int neighbors = GNode.out_edges.size();
      float old_residual = GNode.residual;
      GNode.residual = 0.0; // TODO: does old residual goes to zero
      GNode.rank += old_residual;
      float delta = old_residual*alpha/neighbors;

      if (delta > 0) {

        // Bulk Syncronous for all the nodes
        for (auto& nb : GNode.out_edges) {
          int dest_host = get_host(nb);
          int index;
          if(dest_host == host_ID) {
            index = nb - host_ID*chunk_size;
            graph_nodes[index].residual += delta;
          }
          else
          {

            index = (nb - dest_host*chunk_size);

            // Only find if it's there. May save some time.
            if(host_bitset[dest_host][index] == 1){
              auto ii =  find_if_index(send_buffer[dest_host], index);
              if(ii != send_buffer[dest_host].end()) {
                ii+= 1; // this is delta value corresponding to the index.
                (*ii) += delta;
              }
            }
            else {
              send_buffer[dest_host].push_back(index);
              send_buffer[dest_host].push_back(delta);
              host_bitset[dest_host][index] = 1;
            }
          }
        }
      }
    }

    // TODO: DO ALL-TO-ALL to send sizes around. Transpose. Try an example first.

    //MPI_Alltoall to get send around count from each process to each process.
    // fill receive count. ith count is data to be send to ith process.

    for (int i = 0; i < num_hosts; ++i) {
      send_count[i] = (int)send_buffer[i].size();
      //cout << host_ID << " : to : " << i << " => " << send_count[i] << "\n";
    }

    // Exchange sizes
    int status;
    status = MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    if(status != MPI_SUCCESS) {
      cout << "MPI_Alltoall failed\n";
      exit(0);
    }

    for (int i = 0; i < num_hosts; ++i) {
      //cout  << host_ID << " : from : " << i << " => " << recv_count[i] << "\n";
      recv_buffer[i].resize(recv_count[i]);
    }

    //MPI send and receive messages (for each host)

    // Allocate receive buffer.
    int err;
    for(int dest_host = 0 ; dest_host < num_hosts; ++dest_host) {
      MPI_Status status_sendrecv;
      if(host_ID != dest_host) {
        err = MPI_Sendrecv(&send_buffer[dest_host].front(), send_count[dest_host], MPI_FLOAT, dest_host, DELTA, &recv_buffer[dest_host].front(), recv_count[dest_host], MPI_FLOAT, dest_host, DELTA, MPI_COMM_WORLD, &status_sendrecv);

        if(err != MPI_SUCCESS) {
          cout << "MPI_Sendrecv failed\n";
          exit(0);
        }

        //Apply received delta.
        for(int i = 0; i < num_hosts; ++i) {
          for(int j = 0; j < recv_count[dest_host]; ++j) {
            int i = recv_buffer[dest_host][j] ;
            ++j; // to access delta corresponding to index at j.
            if(host_ID == 0)
              //cout << "RESIDULE : " << recv_buffer[dest_host][j] <<"\n";
            graph_nodes[i].residual += recv_buffer[dest_host][j];
          }
        }

      }
    }


  }
  cout << host_ID << " : RANK : " << graph_nodes[0].rank << "\n";


}


void PageRank_Bulk_AlltoAll(std::vector<Node>& graph_nodes, int chunk_size, int iterations, size_t total_nodes) {

  int num_hosts;
  MPI_Comm_size(MPI_COMM_WORLD, &num_hosts);

  int host_ID;
  MPI_Comm_rank(MPI_COMM_WORLD, &host_ID);

  cout << "Inside function : "<< host_ID <<"\n";
  int send_count[num_hosts];
  int recv_count[num_hosts];

  cout << "Allocating send_buffer : "<< host_ID <<" total nodes : " << total_nodes << "\n";
  //vector<float>send_buffer(total_nodes, 0.0);
  float send_buffer[total_nodes];
  memset(send_buffer, 0.0, total_nodes);

  cout << "Allocating receive_buffer : "<< host_ID <<"\n";
  float receive_buffer[total_nodes];
  memset(receive_buffer, 0.0, total_nodes);

  //vector<float>receive_buffer(total_nodes, 0.0);
  //cout << " Total size(MB) : " << sizeof(int)*send_buffer.size()/(1024*1024) << "\n";

  cout << "Vectors Allocated \n";
  if(host_ID == 0) {
       cout << " BEFORE\n";
      for(int j = 0; j < chunk_size; ++j) {
        cout << "SEND RESIDUAL : " << send_buffer[j] <<"\n";
      }
    }


  //for(int iter = 0; iter < iterations; ++iter) {
    //cout << "Iteration : " << iter << " On host : " << host_ID << "\n";
    //for(auto& GNode : graph_nodes) {

      //int neighbors = GNode.out_edges.size();
      //float old_residual = GNode.residual;
      //GNode.residual = 0.0; // TODO: does old residual goes to zero
      //GNode.rank += old_residual;
      //float delta = old_residual*alpha/neighbors;

      //if (delta > 0) {

        //// Bulk Syncronous for all the nodes
        //for (auto& nb : GNode.out_edges) {
          ////int dest_host = get_host(nb);
          ////int index;
          ////if(dest_host == host_ID) {
            ////index = nb - host_ID*chunk_size;
            ////graph_nodes[index].residual += delta;
          ////}
          ////else
          ////{
            //send_buffer[nb] += delta;
          ////}
        //}
      //}
    //}

    //cout << "One node processed\n";
    //// TODO: DO ALL-TO-ALL to send sizes around. Transpose. Try an example first.

    //// MPI_Alltoall to get send around count from each process to each process.
    //// fill receive count. ith count is data to be send to ith process.

    //// Exchange send and receive buffers
    //// Allocate receive buffer.

    //float *receive_buffer;
    //receive_buffer = (float*)malloc(total_nodes*sizeof(float));
    //if(receive_buffer == NULL)
    //{
      //cout << " Allocating receive buffer failed on host : " << host_ID << "\n";
      //exit(1);
    //}
    //int status;
    //status = MPI_Alltoall(&send_buffer.front(), chunk_size, MPI_FLOAT, receive_buffer, chunk_size, MPI_INT, MPI_COMM_WORLD);
    //if(status != MPI_SUCCESS) {
      //cout << "MPI_Alltoall failed\n";
      //exit(0);
    //}

    //cout << "MPI_All_to_all done\n";
    //// Apply received delta.
    //int index = 0;
    //for(int j = 0; j < total_nodes; ++j) {
      ////if(host_ID == 0){
        ////cout << "index : "<< index << "\n";
        ////cout << "RESIDULE : " << receive_buffer[j] <<"\n";
      ////}
      //graph_nodes[index].residual += receive_buffer[j];
      //index++;
      //if(index >= chunk_size)
        //index = 0;
    //}

    //cout << "residual applied once\n";

  //}


  cout << host_ID << " : RANK : " << graph_nodes[0].rank << "\n";


}


void PageRank(std::vector<Node>& graph_nodes) {

    int host_ID;
  MPI_Comm_rank(MPI_COMM_WORLD, &host_ID);

  std::vector<std::vector<float>> send_buffer;
    for(auto& GNode : graph_nodes) {
      int neighbors = GNode.out_edges.size();
      float old_residual = GNode.residual;
      GNode.residual = 0.0; // TODO: does old residual goes to zero
      GNode.rank += old_residual;
      float delta = old_residual*alpha/neighbors;

      for (auto& nb : GNode.out_edges) {
        //cout << " nb :" << nb  << " getHost : " << get_host(nb) << "\n";
        if(get_host(nb) == host_ID) {
           int i = nb - host_ID*global_chunk_size;
           //cout << "Host : " << host_ID <<"nb : " << nb << " i :" << i <<"\n";
           graph_nodes[i].rank += delta;
        }
        else
        {
          // MPI send delta across network
          int count = 2;
          int send_buffer[count];
          send_buffer[0] = nb;
          send_buffer[1] = delta;
          MPI_Request request;
          MPI_Send(&send_buffer, count, MPI_INT, get_host(nb),DELTA, MPI_COMM_WORLD);
        }
      }
      // Receive msgs
      MPI_Status status;
      int recv_count = 0;
      int flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, DELTA, MPI_COMM_WORLD, &flag, &status);
      if(flag) {
        while(true) {
          MPI_Get_count(&status, MPI_INT, &recv_count);
          int recv_buffer[recv_count];
          MPI_Recv(&recv_buffer, recv_count, MPI_INT, MPI_ANY_SOURCE, DELTA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          //Apply received delta.
          int i = recv_buffer[0] - host_ID*global_chunk_size;
          graph_nodes[i].residual += recv_buffer[1];
          flag = 0;
          MPI_Iprobe(MPI_ANY_SOURCE, DELTA, MPI_COMM_WORLD, &flag, &status);
          //cout << " recieved for node :" << recv_buffer[0] << "\n";
          if(!flag)
            break; // no more messages.
        }
      }

    }

    // Receive msgs
    MPI_Status status;
    int recv_count = 0;
    int flag = 0;

    MPI_Iprobe(MPI_ANY_SOURCE, DELTA, MPI_COMM_WORLD, &flag, &status);
    while(flag) {

      cout << "msgs being received\n";
      MPI_Get_count(&status, MPI_INT, &recv_count);
      int recv_buffer[recv_count];
      MPI_Recv(&recv_buffer, recv_count, MPI_INT, MPI_ANY_SOURCE, DELTA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      //Apply received delta.
      int i = recv_buffer[0] - host_ID*global_chunk_size;
      graph_nodes[i].residual += recv_buffer[1];
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, DELTA, MPI_COMM_WORLD, &flag, &status);


    }

    cout << "RANK : " << graph_nodes[0].rank << "\n";


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

  std::vector<Node> graph_nodes;

  cout << " Calling ReadFile : "<< host_ID << "\n";
  size_t total_nodes = ReadFile(inputFile, graph_nodes);
  //cout << " Host : " << host_ID << "\n";
  //cout << " Graph Construction\n";

  cout << " Host : " << host_ID << " Graph nodes : " << graph_nodes.size() << "\n";

  cout << " Calling PageRank_Bulk_AlltoAll : "<< host_ID << "\n";

  /* Marks the beginning of MPI process */
  MPI_Barrier(MPI_COMM_WORLD);
  start_time = MPI_Wtime();

  //PageRank_Bulk_AlltoAll(graph_nodes, global_chunk_size, iterations, total_nodes);

  MPI_Barrier(MPI_COMM_WORLD);
  end_time = MPI_Wtime();


  MPI_Finalize();

  if(host_ID == 0) {
    cout << "Runtime : " << end_time - start_time << "Seconds \n";
  }
  return 0;
}

