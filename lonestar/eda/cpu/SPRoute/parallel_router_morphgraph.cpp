#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <queue>
#include <limits>
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/AtomicHelpers.h"
#include "galois/runtime/Profile.h"

#include "galois/LargeArray.h"

#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"
#include "Lonestar/BFS_SSSP.h"
using namespace std;

#define VIA_COST 1
float HEIGHT      = 6.0;
float K           = 0.5;
float HEIGHT_STEP = 2.0;
float K_STEP      = 0.05;

float slope = 3.0;

static const int DIST_INFINITY   = std::numeric_limits<int>::max() / 2 - 1;
static const int FIFO_CHUNK_SIZE = 64;
static const int MAZE_CHUNK_SIZE = 32;
static const int OBIM_delta      = 4;

struct Info {
  int grid_x, grid_y, num_layers;
  int* v_capacity;
  int* h_capacity;
  int* min_width;
  int* min_spacing;
  int* via_spacing;
  int lower_left_x, lower_left_y, tile_height, tile_width;
  int num_nets;
  int num_capa_adjust;
  int num_tiles;
};

struct Pin {
  int x, y, layer;
  int tile_x, tile_y;
  Pin() : x(0), y(0), layer(0) {}
};

struct Path {
  int src_tile_x, src_tile_y, src_layer;
  int dst_tile_x, dst_tile_y, dst_layer;
  Path()
      : src_tile_x(-1), src_tile_y(-1), src_layer(-1), dst_tile_x(-1),
        dst_tile_y(-1), dst_layer(-1) {}
  Path(int src_tile_x, int src_tile_y, int src_layer, int dst_tile_x,
       int dst_tile_y, int dst_layer)
      : src_tile_x(src_tile_x), src_tile_y(src_tile_y), src_layer(src_layer),
        dst_tile_x(dst_tile_x), dst_tile_y(dst_tile_y), dst_layer(dst_layer) {}
};

struct Node {
  // galois::substrate::SimpleLock lock;
  float dist;
  bool in_direction[4]; // four directions indicating x+1, y+1, x-1, y-1
  int pin_layer;
  Node() : dist(DIST_INFINITY) {
    pin_layer = 0;
    for (int i = 0; i < 4; i++)
      in_direction[i] = 0;
  }
  inline void reset() {
    pin_layer = 0;
    dist      = DIST_INFINITY;
    for (int i = 0; i < 4; i++)
      in_direction[i] = 0;
  }
  inline bool update_direction(float new_dist, float old_dist, int direction) {
    if (new_dist < old_dist) {
      for (int i = 0; i < 4; i++)
        in_direction[i] = (i == direction) ? 1 : 0;
      return true;
    } else if (new_dist == old_dist) {
      in_direction[direction] = 1;
      return false;
    } else {
      cout << "error in update direction" << endl;
      return false;
    }
  }
  /*inline void acquireLock()
  {
      lock.lock();
  }
  inline void releaseLock()
  {
      lock.unlock();
  } */
};
struct Edge {
  int capacity;
  int utilization;
  float scale;
  float cost;
  Edge() : capacity(0), utilization(0), scale(1), cost(1) {}
  void compute_cost() {
    cost = 1 + HEIGHT / (1 + exp(-K * (float)(utilization - capacity)));
    /*if(utilization > capacity)
        cost += HEIGHT/ slope * ((float)(utilization - capacity));*/
  }
};

using edgeArray = galois::LargeArray<Edge>;
using nodeArray = galois::LargeArray<Node>;

struct Net {
  string name;
  int id;
  int num_pins;
  int min_width;
  Pin* pinlist;
  bool reroute;
  std::list<Path> pathlist;
  std::list<Edge*> edgelist;
  std::list<Node*> nodelist;
};

struct timerstruct {
  galois::StatTimer t_init_dist;
  galois::StatTimer t_trace_back;
  galois::StatTimer t_rebuild_queue;
  galois::StatTimer t_maze_route;
  galois::StatTimer t_print_path;
  galois::StatTimer t_record_path;
  galois::StatTimer t_mazeloop;
  timerstruct()
      : t_init_dist("init_dist"), t_trace_back("trace_back"),
        t_rebuild_queue("rebuild_queue"), t_maze_route("maze_route"),
        t_print_path("print_path"), t_record_path("record_path"),
        t_mazeloop("mazeloop") {}
};

using netArray = galois::LargeArray<Net>;
using Graph    = galois::graphs::MorphGraph<int, void, true>;
using GNode    = Graph::GraphNode;

void init_dist(const Info info, nodeArray& nodelist) {
  galois::on_each([&](const unsigned tid, const unsigned numT) {
    unsigned total_nodes = (info.grid_x - 1) * (info.grid_y - 1);
    unsigned start       = total_nodes / numT * tid;
    unsigned end =
        (tid == numT - 1) ? total_nodes : total_nodes / numT * (tid + 1);
    for (unsigned i = start; i < end; i++)
      nodelist[i].reset();
  });

  /*for(int y = 0; y < info.grid_y - 1; y++)
  {
      for(int x = 0; x < info.grid_x - 1; x++)
      {
          nodelist[y * (info.grid_x - 1) + x].reset();
      }
  }*/
}

void trace_back(const Info info, nodeArray& nodelist, int nodeid, int netid) {
  Node* dst = &nodelist[nodeid];
  int srcid;
  int dstid          = nodeid;
  int prev_direction = -1;
  while (dst->dist != 0) {
    int tile_y = dstid / (info.grid_x - 1);
    int tile_x = dstid % (info.grid_x - 1);
    // cout << "trace back:"<< dstid << "("<< tile_x << "," << tile_y << ") " <<
    // "dist: " << nodelist[dstid].dist << " "; cout << dst->in_direction[0] <<
    // dst->in_direction[1] << dst->in_direction[2] <<
    // dst->in_direction[3]<<endl;

    int rand_offset   = (prev_direction >= 0) ? prev_direction : rand() % 4;
    int end_direction = -1;
    for (int i = 0; i < 4; i++) {
      if (dst->in_direction[(i + rand_offset) % 4]) {
        end_direction  = (i + rand_offset) % 4;
        prev_direction = end_direction;
        break;
      }
    }
    if (end_direction >= 0) {
      for (int i = 0; i < 4; i++) {
        dst->in_direction[i] = (i == end_direction) ? 1 : 0;
      }
    } else {
      cout << " error in end direction reset" << endl;
      exit(1);
    }

    switch (end_direction) {
    case 0:
      srcid = dstid + 1;
      break;
    case 1:
      srcid = dstid + info.grid_x - 1;
      break;
    case 2:
      srcid = dstid - 1;
      break;
    case 3:
      srcid = dstid - (info.grid_x - 1);
      break;
    default: {

      cout << "error in trace back direction: end_direction = " << end_direction
           << " tile: " << tile_x << "," << tile_y;
      cout << " netid: " << netid << endl;

      cout << "direction:" << dst->in_direction[0]
           << " distance: " << nodelist[dstid + 1].dist << endl;
      cout << "direction:" << dst->in_direction[1]
           << " distance: " << nodelist[dstid + info.grid_x - 1].dist << endl;
      cout << "direction:" << dst->in_direction[2]
           << " distance: " << nodelist[dstid - 1].dist << endl;
      cout << "direction:" << dst->in_direction[3]
           << " distance: " << nodelist[dstid - (info.grid_x - 1)].dist << endl;

      // exit(1);
    }
    }

    if (srcid < 0 || srcid > info.num_tiles - 1) {
      cout << "error in trace back srcid" << endl;
      // exit(1);
    }
    nodelist[dstid].dist = 0;
    dstid                = srcid;
    dst                  = &nodelist[dstid];
  }
}

void rebuild_queue(
    const Info info, nodeArray& nodelist, galois::InsertBag<int>& allNodeBag,
    galois::InsertBag<int>& initBag,
    bool last_pin) // actually this src is the dst just connected to the net
{
  /*galois::do_all(galois::iterate(allNodeBag),
      [&] (const auto nodeid)
      {
              int neighborid;
              if(nodelist[nodeid].dist == 0)
              {
                  //cout<<"rebuild queue: ("<< nodeid %(info.grid_x - 1) << ","
     << nodeid/(info.grid_x - 1) <<")"<< endl; neighborid = nodeid + 1;
                  if(neighborid <= info.num_tiles - 1 &&
     nodelist[neighborid].dist == 0 && nodelist[neighborid].in_direction[2])
                  {
                      nodelist[nodeid].in_direction[0] = 1;
                  }

                  neighborid = nodeid - 1;
                  if(neighborid >= 0 && nodelist[neighborid].dist == 0 &&
     nodelist[neighborid].in_direction[0])
                  {
                      nodelist[nodeid].in_direction[2] = 1;
                  }

                  neighborid = nodeid + info.grid_x - 1;
                  if(neighborid <= info.num_tiles - 1 &&
     nodelist[neighborid].dist == 0 && nodelist[neighborid].in_direction[3])
                  {
                      nodelist[nodeid].in_direction[1] = 1;
                  }

                  neighborid = nodeid - (info.grid_x - 1);
                  if(neighborid >= 0 && nodelist[neighborid].dist == 0 &&
     nodelist[neighborid].in_direction[1])
                  {
                      nodelist[nodeid].in_direction[3] = 1;
                  }
                  if(!last_pin)
                      initBag.push(nodeid);

              }
              else
              {
                  nodelist[nodeid].reset();
              }

      },
      galois::steal(),
      galois::chunk_size<2048>()
      );*/
  galois::on_each([&](const unsigned tid, const unsigned numT) {
    unsigned total_nodes = (info.grid_x - 1) * (info.grid_y - 1);
    unsigned start       = total_nodes / numT * tid;
    unsigned end =
        (tid == numT - 1) ? total_nodes : total_nodes / numT * (tid + 1);

    for (unsigned nodeid = start; nodeid < end; nodeid++) {
      int neighborid;
      if (nodelist[nodeid].dist == 0) {
        // cout<<"rebuild queue: ("<< nodeid %(info.grid_x - 1) << "," <<
        // nodeid/(info.grid_x - 1) <<")"<< endl;
        neighborid = nodeid + 1;
        if (neighborid <= info.num_tiles - 1 &&
            nodelist[neighborid].dist == 0 &&
            nodelist[neighborid].in_direction[2]) {
          nodelist[nodeid].in_direction[0] = 1;
        }

        neighborid = nodeid - 1;
        if (neighborid >= 0 && nodelist[neighborid].dist == 0 &&
            nodelist[neighborid].in_direction[0]) {
          nodelist[nodeid].in_direction[2] = 1;
        }

        neighborid = nodeid + info.grid_x - 1;
        if (neighborid <= info.num_tiles - 1 &&
            nodelist[neighborid].dist == 0 &&
            nodelist[neighborid].in_direction[3]) {
          nodelist[nodeid].in_direction[1] = 1;
        }

        neighborid = nodeid - (info.grid_x - 1);
        if (neighborid >= 0 && nodelist[neighborid].dist == 0 &&
            nodelist[neighborid].in_direction[1]) {
          nodelist[nodeid].in_direction[3] = 1;
        }
        if (!last_pin)
          initBag.push(nodeid);

      } else {
        nodelist[nodeid].reset();
      }
    }
  });

  /*for(int nodeid = 0; nodeid < info.num_tiles; nodeid++)
  {
      int neighborid;
      if(nodelist[nodeid].dist == 0)
      {
          //cout<<"rebuild queue: ("<< nodeid %(info.grid_x - 1) << "," <<
  nodeid/(info.grid_x - 1) <<")"<< endl; neighborid = nodeid + 1; if(neighborid
  <= info.num_tiles - 1 && nodelist[neighborid].dist == 0 &&
  nodelist[neighborid].in_direction[2])
          {
              nodelist[nodeid].in_direction[0] = 1;
          }

          neighborid = nodeid - 1;
          if(neighborid >= 0 && nodelist[neighborid].dist == 0 &&
  nodelist[neighborid].in_direction[0])
          {
              nodelist[nodeid].in_direction[2] = 1;
          }

          neighborid = nodeid + info.grid_x - 1;
          if(neighborid <= info.num_tiles - 1 && nodelist[neighborid].dist == 0
  && nodelist[neighborid].in_direction[3])
          {
              nodelist[nodeid].in_direction[1] = 1;
          }

          neighborid = nodeid - (info.grid_x - 1);
          if(neighborid >= 0 && nodelist[neighborid].dist == 0 &&
  nodelist[neighborid].in_direction[1])
          {
              nodelist[nodeid].in_direction[3] = 1;
          }
          if(!last_pin)
              initBag.push(nodeid);

      }
      else
      {
          nodelist[nodeid].reset();
      }
  }*/
}

inline int horizontal_min_spacing(const Info info, edgeArray& h_edge,
                                  int nodeid) {
  for (int i = 0; i < info.num_layers; i++) {
    // if(v_edge[nodeid].utilization < v_edge[nodeid].capacity)
    if (info.h_capacity[i] != 0)
      return info.min_spacing[i];
  }
  cout << "error in finding horizontal min spacing!" << endl;
  exit(1);
}

inline int vertical_min_spacing(const Info info, edgeArray& v_edge,
                                int nodeid) {
  for (int i = 0; i < info.num_layers; i++) {
    // if(v_edge[nodeid].utilization < v_edge[nodeid].capacity)
    if (info.h_capacity[i] != 0)
      return info.min_spacing[i];
  }
  cout << "error in finding horizontal min spacing!" << endl;
  exit(1);
}

inline int horizontal_layer(const Info info, edgeArray& h_edge, int nodeid) {
  for (int i = 0; i < info.num_layers; i++) {
    // if(v_edge[nodeid].utilization < v_edge[nodeid].capacity)
    if (info.h_capacity[i] != 0)
      return i + 1;
  }
  cout << "error in finding horizontal layer!" << endl;
  exit(1);
}

inline int vertical_layer(const Info info, edgeArray& v_edge, int nodeid) {
  for (int i = 0; i < info.num_layers; i++) {
    if (info.v_capacity[i] != 0)
      return i + 1;
  }
  cout << "error in finding horizontal layer!" << endl;
  exit(1);
}

void record_path(const Info info, nodeArray& nodelist, edgeArray& h_edge,
                 edgeArray& v_edge, Net& net) {
  for (int nodeid = 0; nodeid < info.num_tiles; nodeid++) {
    int neighborid;
    Node& node = nodelist[nodeid];
    if (nodelist[nodeid].dist == 0) {
      // cout<< nodeid << " " << nodeid % (info.grid_x - 1) << " " << nodeid
      // /(info.grid_x - 1) << " " << node.in_direction[0] <<
      // node.in_direction[1] << node.in_direction[2] << node.in_direction[3] <<
      // endl;
      neighborid = nodeid + 1;
      if (nodeid % (info.grid_x - 1) != info.grid_x - 2 &&
          nodelist[neighborid].dist == 0 && node.in_direction[0] &&
          nodelist[neighborid].in_direction[2])
      //&& h_edge[nodeid].utilization + net.min_width +
      // horizontal_min_spacing(info, h_edge, nodeid) <=
      // h_edge[nodeid].capacity)
      {
        Path* path       = new Path;
        path->src_tile_x = nodeid % (info.grid_x - 1);
        path->src_tile_y = nodeid / (info.grid_x - 1);
        path->dst_tile_x = nodeid % (info.grid_x - 1) + 1;
        path->dst_tile_y = nodeid / (info.grid_x - 1);

        path->src_layer = horizontal_layer(info, h_edge, nodeid);
        path->dst_layer = path->src_layer;

        h_edge[nodeid].utilization +=
            net.min_width + horizontal_min_spacing(info, h_edge, nodeid);
        h_edge[nodeid].compute_cost();
        /*h_edge[nodeid].cost = 1 + HEIGHT / ( 1 + exp(- K *
        (h_edge[nodeid].utilization - h_edge[nodeid].capacity)));
        if(h_edge[nodeid].utilization > h_edge[nodeid].capacity)
            h_edge[nodeid].cost += HEIGHT/ slope *
        ((float)(h_edge[nodeid].utilization - h_edge[nodeid].capacity));*/
        net.pathlist.push_back(*path);
        net.edgelist.push_back(&h_edge[nodeid]);
        /*{
            cout<< "utilization exceeds capacity in h net " << net.id << " node:
        " << path->src_tile_x << "," << path->src_tile_y << endl; cout << " "<<
        h_edge[nodeid].utilization << "/" << h_edge[nodeid].capacity << endl;
            exit(1);
        }*/
      }

      neighborid = nodeid + info.grid_x - 1;
      if (neighborid <= info.num_tiles - 1 && nodelist[neighborid].dist == 0 &&
          node.in_direction[1] && nodelist[neighborid].in_direction[3])
      //&& v_edge[nodeid].utilization + net.min_width +
      // vertical_min_spacing(info, v_edge, nodeid) <= v_edge[nodeid].capacity)
      {
        Path* path       = new Path;
        path->src_tile_x = nodeid % (info.grid_x - 1);
        path->src_tile_y = nodeid / (info.grid_x - 1);
        path->dst_tile_x = nodeid % (info.grid_x - 1);
        path->dst_tile_y = nodeid / (info.grid_x - 1) + 1;

        path->src_layer = vertical_layer(info, v_edge, nodeid);
        path->dst_layer = path->src_layer;

        v_edge[nodeid].utilization +=
            net.min_width + vertical_min_spacing(info, v_edge, nodeid);
        v_edge[nodeid].compute_cost();
        /*v_edge[nodeid].cost = 1 + HEIGHT / ( 1 + exp(- K *
        (v_edge[nodeid].utilization - v_edge[nodeid].capacity)));
        if(v_edge[nodeid].utilization > v_edge[nodeid].capacity)
            v_edge[nodeid].cost += HEIGHT/ slope *
        ((float)(v_edge[nodeid].utilization - v_edge[nodeid].capacity));*/
        net.pathlist.push_back(*path);
        net.edgelist.push_back(&v_edge[nodeid]);

        /*{
            cout<< "utilization exceeds capacity in v net " << net.id << " node:
        " << path->src_tile_x << "," << path->src_tile_y << endl; cout << " "<<
        v_edge[nodeid].utilization << "/" << v_edge[nodeid].capacity << endl;
            exit(1);
        }*/
      }

      if ((nodelist[nodeid].in_direction[0] ||
           nodelist[nodeid].in_direction[2] ||
           nodelist[nodeid].pin_layer ==
               horizontal_layer(info, h_edge, nodeid)) &&
          (nodelist[nodeid].in_direction[1] ||
           nodelist[nodeid].in_direction[3] ||
           nodelist[nodeid].pin_layer ==
               vertical_layer(info, v_edge, nodeid))) {
        Path* path       = new Path;
        path->src_tile_x = nodeid % (info.grid_x - 1);
        path->src_tile_y = nodeid / (info.grid_x - 1);
        path->dst_tile_x = nodeid % (info.grid_x - 1);
        path->dst_tile_y = nodeid / (info.grid_x - 1);

        path->src_layer = 1;
        path->dst_layer = 2; // only for 2D

        net.pathlist.push_back(*path);
        net.nodelist.push_back(&nodelist[nodeid]);
      }
    } else
      nodelist[nodeid].reset();
  }
}
void printnode(const Info info, nodeArray& nodelist, int node_x, int node_y) {
  int nodeid = node_x + (info.grid_x - 1) * node_y;
  Node& node = nodelist[nodeid];
  cout << "node: " << nodeid << "(" << node_x << "," << node_y << ") "
       << node.in_direction[0] << node.in_direction[1] << node.in_direction[2];
  cout << node.in_direction[3] << endl;
}

void print_capacity(const Info info, edgeArray& v_edge, edgeArray& h_edge) {
  for (int i = 0; i < (info.grid_x - 1) * (info.grid_y - 1); i++) {
    if (v_edge[i].utilization > 0)
      cout << "utilization at v edge: (" << i % (info.grid_x - 1) << ","
           << i / (info.grid_x - 1) << ") " << v_edge[i].utilization << endl;
  }
  for (int i = 0; i < (info.grid_x - 1) * (info.grid_y - 1); i++) {
    if (h_edge[i].utilization > 0)
      cout << "utilization at h edge: (" << i % (info.grid_x - 1) << ","
           << i / (info.grid_x - 1) << ") " << h_edge[i].utilization << endl;
  }
}

struct UpdateRequestIndexer {
  nodeArray& nodelist;

  unsigned int operator()(const int& req) const {
    unsigned int t = (unsigned int)(nodelist[req].dist) / OBIM_delta;
    return t;
  }
};

void mazeroute(const Info info, nodeArray& nodelist,
               galois::InsertBag<int>& allNodeBag, edgeArray& v_edge,
               edgeArray& h_edge, Net* netlist, Graph& g,
               std::vector<GNode>& node_GNode, timerstruct& timer) {
  // std::queue<int> node_queue; // store nodeid
  // int reference_dist;
  std::atomic<float> reference_dist;

  namespace gwl = galois::worklists;
  using dChunk  = gwl::ChunkFIFO<FIFO_CHUNK_SIZE>;
  using OBIM    = gwl::OrderedByIntegerMetric<UpdateRequestIndexer, dChunk>;

  int num_reroute     = 0;
  int sweep_direction = rand() % 2;
  for (int netid = rand() % (info.num_nets), cnt = 0; cnt < info.num_nets;
       cnt++) {
    if (sweep_direction)
      netid = (netid + 1) % info.num_nets;
    else
      netid = (netid == 0) ? (info.num_nets - 1) : (netid - 1);

    if (netid % 10000 == 0)
      cout << "net: " << netid << " " << netlist[netid].id << endl;
    Net& net = netlist[netid];
    if (net.num_pins > 1000 ||
        !net.reroute) // contest says no need to route it for now, but we have
                      // to resolve it
      continue;
    num_reroute++;
    /*if(net.min_width != 1)
        cout<<"net: " << net.id << " minwidth: " << net.min_width << endl;*/
    bool debug = false;
    /*if(netid >= 20000)
    {
        cout<<"remove from netid"<<net.id<<endl;
        break;
    }*/
    /*if(netid == 11285)
        debug = true;*/
    timer.t_init_dist.start();
    init_dist(info, nodelist);
    timer.t_init_dist.stop();

    bool last_pin = false;
    galois::InsertBag<int> initBag;
    for (int pinid = 0; pinid < net.num_pins; pinid++) {
      Pin& pin = net.pinlist[pinid];

      int nodeid     = pin.tile_y * (info.grid_x - 1) + pin.tile_x;
      Node& node     = nodelist[nodeid];
      node.pin_layer = pin.layer;
      if (debug == true)
        cout << "pin: " << pinid << " tile: " << pin.tile_x << " " << pin.tile_y
             << endl;

      reference_dist.store(DIST_INFINITY);

      if (pinid == net.num_pins - 1)
        last_pin = true;

      if (pinid == 0) {
        node.dist = 0;

        if (info.v_capacity[pin.layer - 1] != 0) // vertical direction layer
        {
          node.in_direction[1] = 1;
          node.in_direction[3] = 1;
        } else if (info.h_capacity[pin.layer - 1] !=
                   0) // herizontal direction layer
        {
          node.in_direction[0] = 1;
          node.in_direction[2] = 1;
        } else
          cout << "error in init qnode" << endl;

        // node_queue.push(nodeid);
        initBag.push(nodeid);
        /*int tile_x = nodeid % (info.grid_x - 1);
        int tile_y = nodeid / (info.grid_x - 1);
        cout<<"push pin 0 :" << tile_x << " " << tile_y << endl;*/

      } else if (node.dist != 0) {
        timer.t_mazeloop.start();
        galois::for_each(
            galois::iterate(initBag),
            [&](const auto& srcid, auto& ctx)

            {
              GNode src_GNode = node_GNode[srcid];
              float local_reference_dist =
                  reference_dist; // or reference_dist.load()
              Node& src = nodelist[srcid];

              /*if(debug)
              {
                  int tile_x = srcid % (info.grid_x - 1);
                  int tile_y = srcid / (info.grid_x - 1);
                  cout<<"src: \t"<< tile_x << "\t"<< tile_y <<"\tdistance:\t" <<
              nodelist[srcid].dist << "\tindirection: "; cout<<
              src.in_direction[0] <<
              src.in_direction[1]<<src.in_direction[2]<<src.in_direction[3]<<endl;
                  cout<<"to 0: " << h_edge[srcid].utilization << "/"<<
              h_edge[srcid].capacity <<endl; cout<<"to 1: " <<
              v_edge[srcid].utilization << "/"<< v_edge[srcid].capacity <<endl;
                  cout<<"to 2: " << h_edge[srcid - 1].utilization << "/"<<
              h_edge[srcid - 1].capacity <<endl; cout<<"to 3: " << v_edge[srcid
              - (info.grid_x - 1)].utilization << "/"<< v_edge[srcid -
              (info.grid_x - 1)].capacity <<endl;
              }*/

              // int edge_cnt = 0;
              int dstid;
              for (auto e : g.edges(src_GNode)) {
                auto dst_GNode = g.getEdgeDst(e);
                dstid          = g.getData(dst_GNode);
                if (dstid == srcid + 1) {
                  Node& dst    = nodelist[dstid];
                  Edge& r_edge = h_edge[srcid];

                  if (src.in_direction[2] &&
                      src.dist + r_edge.cost <= dst.dist &&
                      src.dist + r_edge.cost <=
                          local_reference_dist) // previous node is x - 1
                  {
                    bool updated = dst.update_direction(src.dist + r_edge.cost,
                                                        dst.dist, 2);
                    dst.dist     = src.dist + r_edge.cost;

                    if (dstid == nodeid) // if dst is one of the destination
                                         // pins
                      local_reference_dist = dst.dist;
                    else if (updated)
                      ctx.push(dstid);
                  } else if ((src.in_direction[1] || src.in_direction[3]) &&
                             src.dist + r_edge.cost + VIA_COST <= dst.dist &&
                             src.dist + r_edge.cost + VIA_COST <=
                                 local_reference_dist) {
                    bool updated = dst.update_direction(
                        src.dist + r_edge.cost + VIA_COST, dst.dist, 2);
                    dst.dist = src.dist + r_edge.cost + VIA_COST;

                    if (dstid == nodeid) // if dst is one of the destination
                                         // pins
                      local_reference_dist = dst.dist;
                    else if (updated)
                      ctx.push(dstid);
                  }

                }

                else if (dstid == srcid - 1) {
                  Node& dst    = nodelist[dstid];
                  Edge& l_edge = h_edge[srcid - 1];

                  if (src.in_direction[0] &&
                      src.dist + l_edge.cost <= dst.dist &&
                      src.dist + l_edge.cost <=
                          local_reference_dist) // previous node is x + 1
                  {
                    bool updated = dst.update_direction(src.dist + l_edge.cost,
                                                        dst.dist, 0);
                    dst.dist     = src.dist + l_edge.cost;

                    if (dstid == nodeid) // if dst is one of the destination
                                         // pins
                      local_reference_dist = dst.dist;
                    else if (updated)
                      ctx.push(dstid);
                  } else if ((src.in_direction[1] || src.in_direction[3]) &&
                             src.dist + l_edge.cost + VIA_COST <= dst.dist &&
                             src.dist + l_edge.cost + VIA_COST <=
                                 local_reference_dist) {
                    bool updated = dst.update_direction(
                        src.dist + l_edge.cost + VIA_COST, dst.dist, 0);
                    dst.dist = src.dist + l_edge.cost + VIA_COST;

                    if (dstid == nodeid) // if dst is one of the destination
                                         // pins
                      local_reference_dist = dst.dist;
                    else if (updated)
                      ctx.push(dstid);
                  }
                }

                else if (dstid == srcid + info.grid_x - 1) {
                  Node& dst    = nodelist[dstid];
                  Edge& t_edge = v_edge[srcid];

                  if (src.in_direction[3] &&
                      src.dist + t_edge.cost <= dst.dist &&
                      src.dist + t_edge.cost <=
                          local_reference_dist) // previous node is y - 1
                  {
                    bool updated = dst.update_direction(src.dist + t_edge.cost,
                                                        dst.dist, 3);
                    dst.dist     = src.dist + t_edge.cost;

                    if (dstid == nodeid) // if dst is one of the destination
                                         // pins
                      local_reference_dist = dst.dist;
                    else if (updated)
                      ctx.push(dstid);
                  } else if ((src.in_direction[0] || src.in_direction[2]) &&
                             src.dist + t_edge.cost + VIA_COST <= dst.dist &&
                             src.dist + t_edge.cost + VIA_COST <=
                                 local_reference_dist) {
                    bool updated = dst.update_direction(
                        src.dist + t_edge.cost + VIA_COST, dst.dist, 3);
                    dst.dist = src.dist + t_edge.cost + VIA_COST;

                    if (dstid == nodeid) // if dst is one of the destination
                                         // pins
                      local_reference_dist = dst.dist;
                    else if (updated)
                      ctx.push(dstid);
                  }
                } else if (dstid == srcid - (info.grid_x - 1)) {
                  Node& dst    = nodelist[dstid];
                  Edge& b_edge = v_edge[srcid - (info.grid_x - 1)];

                  if (src.in_direction[1] &&
                      src.dist + b_edge.cost <= dst.dist &&
                      src.dist + b_edge.cost <=
                          local_reference_dist) // previous node is y + 1
                  {
                    bool updated = dst.update_direction(src.dist + b_edge.cost,
                                                        dst.dist, 1);
                    dst.dist     = src.dist + b_edge.cost;

                    if (dstid == nodeid) // if dst is one of the destination
                                         // pins
                      local_reference_dist = dst.dist;
                    else if (updated)
                      ctx.push(dstid);
                  } else if ((src.in_direction[0] || src.in_direction[2]) &&
                             src.dist + b_edge.cost + VIA_COST <= dst.dist &&
                             src.dist + b_edge.cost + VIA_COST <=
                                 local_reference_dist) {
                    bool updated = dst.update_direction(
                        src.dist + b_edge.cost + VIA_COST, dst.dist, 1);
                    dst.dist = src.dist + b_edge.cost + VIA_COST;

                    if (dstid == nodeid) // if dst is one of the destination
                                         // pins
                      local_reference_dist = dst.dist;
                    else if (updated)
                      ctx.push(dstid);
                  }
                } else {
                  cout << "there must be something wrong in constructing the "
                          "graph"
                       << endl;
                  exit(1);
                }
              }

              galois::atomicMin(reference_dist, local_reference_dist);
            },
            galois::wl<dChunk>(),
            // galois::wl<OBIM>(UpdateRequestIndexer{nodelist}),
            galois::steal(), galois::chunk_size<MAZE_CHUNK_SIZE>());
        timer.t_mazeloop.stop();

        // while queue is empty, nodeid is reached.
        timer.t_trace_back.start();
        trace_back(info, nodelist, nodeid, net.id);
        if (info.v_capacity[pin.layer - 1] != 0) // vertical direction layer
        {
          node.in_direction[1] = 1;
          node.in_direction[3] = 1;
        } else if (info.h_capacity[pin.layer - 1] !=
                   0) // herizontal direction layer
        {
          node.in_direction[0] = 1;
          node.in_direction[2] = 1;
        } else
          cout << "error in init qnode" << endl;
        timer.t_trace_back.stop();

        timer.t_rebuild_queue.start();
        rebuild_queue(info, nodelist, allNodeBag, initBag, last_pin);
        timer.t_rebuild_queue.stop();
        // printnode(info, nodelist, 99, 191);
        /*if(last_pin && !node_queue.empty())
        {
            cout<<"queue is not empty when finish a net"<<endl;
        }*/
      }
    }
    timer.t_record_path.start();
    record_path(info, nodelist, h_edge, v_edge, net);
    timer.t_record_path.stop();
    // print_capacity(info, v_edge, h_edge);
  }
  cout << "num of routed nets :" << num_reroute << endl;
}

void readfile(ifstream& infile, Info& info, nodeArray& nodelist,
              edgeArray& v_edge, edgeArray& h_edge, Net*& netlist, Graph& g,
              std::vector<GNode>& node_GNode) {
  string temp1, temp2;
  infile >> temp1 >> info.grid_x >> info.grid_y >> info.num_layers;
  // cout << info.grid_x << info.grid_y << info.layer << endl;
  info.num_tiles = (info.grid_x - 1) * (info.grid_y - 1);
  v_edge.allocateInterleaved(info.num_tiles);
  h_edge.allocateInterleaved(info.num_tiles);
  // v_edge.allocateBlocked(info.num_tiles);
  // h_edge.allocateBlocked(info.num_tiles);
  int nlayers      = info.num_layers;
  info.v_capacity  = new int[nlayers];
  info.h_capacity  = new int[nlayers];
  info.min_width   = new int[nlayers];
  info.min_spacing = new int[nlayers];
  info.via_spacing = new int[nlayers];
  infile >> temp1 >> temp2;

  if (temp1 == "vertical" && temp2 == "capacity") {
    for (int i = 0; i < nlayers; i++) {
      infile >> info.v_capacity[i];
      // cout<<info.v_capacity[i]<<" ";
    }
    // cout<<endl;
  }

  infile >> temp1 >> temp2;
  if (temp1 == "horizontal" && temp2 == "capacity") {
    for (int i = 0; i < nlayers; i++) {
      infile >> info.h_capacity[i];
      // cout<<info.h_capacity[i]<<" ";
    }
    // cout<<endl;
  }

  infile >> temp1 >> temp2;
  if (temp1 == "minimum" && temp2 == "width") {
    for (int i = 0; i < nlayers; i++) {
      infile >> info.min_width[i];
      // cout<<info.min_width[i]<<" ";
    }
    // cout<<endl;
  }

  infile >> temp1 >> temp2;
  if (temp1 == "minimum" && temp2 == "spacing") {
    for (int i = 0; i < nlayers; i++) {
      infile >> info.min_spacing[i];
      // cout<<info.min_spacing[i]<<" ";
    }
    // cout<<endl;
  }

  infile >> temp1 >> temp2;
  if (temp1 == "via" && temp2 == "spacing") {
    for (int i = 0; i < nlayers; i++) {
      infile >> info.via_spacing[i];
      // cout<<info.via_spacing[i]<<" ";
    }
    // cout<<endl;
  }

  infile >> info.lower_left_x >> info.lower_left_y >> info.tile_width >>
      info.tile_height;
  // cout << info.lower_left_x << info.lower_left_y << info.tile_width <<
  // info.tile_height << endl;

  infile >> temp1 >> temp2;
  if (temp1 == "num" && temp2 == "net")
    infile >> info.num_nets;

  netlist = new Net[info.num_nets];
  for (int i = 0; i < info.num_nets; i++) {
    infile >> netlist[i].name >> netlist[i].id >> netlist[i].num_pins >>
        netlist[i].min_width;
    int num_pins       = netlist[i].num_pins;
    netlist[i].pinlist = new Pin[num_pins];
    netlist[i].reroute = true;
    for (int j = 0; j < netlist[i].num_pins; j++) {
      infile >> netlist[i].pinlist[j].x >> netlist[i].pinlist[j].y >>
          netlist[i].pinlist[j].layer;
      netlist[i].pinlist[j].tile_x =
          (netlist[i].pinlist[j].x - info.lower_left_x) / info.tile_width;
      netlist[i].pinlist[j].tile_y =
          (netlist[i].pinlist[j].y - info.lower_left_y) / info.tile_height;
    }
  }

  infile >> info.num_capa_adjust;
  int src_tile_x, src_tile_y, dst_tile_x, dst_tile_y, src_layer, dst_layer,
      capacity;
  int total_v_capacity = 0, total_h_capacity = 0;
  for (int i = 0; i < info.num_layers; i++) {
    total_v_capacity += info.v_capacity[i];
    total_h_capacity += info.h_capacity[i];
  }
  for (int i = 0; i < (info.grid_x - 1) * (info.grid_y - 2); i++)
    v_edge[i].capacity = total_v_capacity;

  for (int i = 0; i < (info.grid_x - 1) * (info.grid_y - 1); i++)
    if (i % (info.grid_x - 1) != info.grid_x - 2)
      h_edge[i].capacity = total_h_capacity;

  for (int i = 0; i < info.num_capa_adjust; i++) {
    infile >> src_tile_x >> src_tile_y >> src_layer;
    infile >> dst_tile_x >> dst_tile_y >> dst_layer;
    infile >> capacity;
    if (dst_tile_x - src_tile_x == 1)
      h_edge[src_tile_y * (info.grid_x - 1) + src_tile_x].capacity = capacity;
    else if (dst_tile_y - src_tile_y == 1)
      v_edge[src_tile_y * (info.grid_x - 1) + src_tile_x].capacity = capacity;
    else {
      cout << "error in reading capacity adjustment" << endl;
      exit(1);
    }
  }

  nodelist.allocateInterleaved(info.num_tiles);
  node_GNode.resize(info.num_tiles);
  for (int i = 0; i < info.num_tiles; i++) {
    GNode n = g.createNode(i);
    g.addNode(n);
    node_GNode[i] = n;
  }

  for (int nodeid = 0; nodeid < info.num_tiles; nodeid++) {
    GNode src = node_GNode[nodeid];
    int neighborid;
    neighborid = nodeid + 1;
    if (nodeid % (info.grid_x - 1) != info.grid_x - 2) {
      GNode dst = node_GNode[neighborid];
      g.addEdge(src, dst);
      // g.addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED );
    }

    neighborid = nodeid - 1;
    if (nodeid % (info.grid_x - 1) != 0) {
      GNode dst = node_GNode[neighborid];
      g.addEdge(src, dst);
      // g.addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED );
    }

    neighborid = nodeid + (info.grid_x - 1);
    if (neighborid <= info.num_tiles - 1) {
      GNode dst = node_GNode[neighborid];
      g.addEdge(src, dst);
      // g.addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED );
    }

    neighborid = nodeid - (info.grid_x - 1);
    if (neighborid >= 0) {
      GNode dst = node_GNode[neighborid];
      g.addEdge(src, dst);
      // g.addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED );
    }
  }

  infile.close();
}

void print_path(const Info info, Net* netlist) {
  ofstream outfile("router.out");
  for (int netid = 0; netid < info.num_nets; netid++) {
    Net& net = netlist[netid];
    // if(netid != 2)
    //    continue;
    outfile << net.name << " " << net.id << endl;

    for (std::list<Path>::iterator path = net.pathlist.begin();
         path != net.pathlist.end(); path++) {
      // cout<< tile_x<<" " << tile_y << endl;
      int src_cor_x = path->src_tile_x * info.tile_width +
                      0.5 * info.tile_width + info.lower_left_x;
      int src_cor_y = path->src_tile_y * info.tile_height +
                      0.5 * info.tile_height + info.lower_left_y;
      int src_layer = path->src_layer;

      int dst_cor_x = path->dst_tile_x * info.tile_width +
                      0.5 * info.tile_width + info.lower_left_x;
      int dst_cor_y = path->dst_tile_y * info.tile_height +
                      0.5 * info.tile_height + info.lower_left_y;
      int dst_layer = path->dst_layer;

      // outfile<< "(" << path->src_tile_x << "," << path->src_tile_y << "," <<
      // src_layer << ")-(" << path->dst_tile_x << "," << path->dst_tile_y <<
      // ","
      // << dst_layer << ")" << endl;
      outfile << "(" << src_cor_x << "," << src_cor_y << "," << src_layer
              << ")-(" << dst_cor_x << "," << dst_cor_y << "," << dst_layer
              << ")" << endl;
    }
    outfile << "!" << endl;
  }
}

static const char* name = "Single Source Shortest Path";
static const char* desc =
    "Computes the shortest path from a source node to all nodes in a directed "
    "graph using a modified chaotic iteration algorithm";
static const char* url = "single_source_shortest_path";

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);

int check_overflow(const Info info, nodeArray& nodelist, edgeArray& v_edge,
                   edgeArray& h_edge, Net* netlist, bool print) {
  int total_of = 0;
  for (int i = 0; i < (info.grid_x - 1) * (info.grid_y - 1); i++) {
    if (i % (info.grid_x - 1) != info.grid_x - 2) {
      if (h_edge[i].utilization <= h_edge[i].capacity) {
        h_edge[i].scale = 1.0;
      } else {
        h_edge[i].scale =
            (float)h_edge[i].utilization / (float)h_edge[i].capacity;
        total_of += h_edge[i].utilization - h_edge[i].capacity;
        h_edge[i].utilization = 0;
      }
      // if(i % (info.grid_x - 1) == 121 && i / (info.grid_x - 1) == 86)
      if (print) {
        cout << "h_edge " << i % (info.grid_x - 1) << " "
             << i / (info.grid_x - 1) << " overflow: " << h_edge[i].utilization
             << "/" << h_edge[i].capacity << " = " << h_edge[i].scale << endl;
      }
    }
  }

  for (int i = 0; i < (info.grid_x - 1) * (info.grid_y - 2); i++) {
    if (v_edge[i].utilization <= v_edge[i].capacity) {
      v_edge[i].scale = 1.0;
    } else {
      v_edge[i].scale =
          (float)v_edge[i].utilization / (float)v_edge[i].capacity;
      total_of += v_edge[i].utilization - v_edge[i].capacity;
      if (print) {
        cout << "v_edge " << i % (info.grid_x - 1) << " "
             << i / (info.grid_x - 1) << " " << i
             << " overflow: " << v_edge[i].utilization << "/"
             << v_edge[i].capacity << " = " << v_edge[i].scale << endl;
      }
      v_edge[i].utilization = 0;
    }
  }

  cout << "total overflow is " << total_of << endl;
  return total_of;
}

void reset_net(const Info info, nodeArray& nodelist, edgeArray& v_edge,
               edgeArray& h_edge, Net* netlist) {
  int num_reroute = 0;
  for (int i = 0; i < info.num_nets; i++) {
    bool no_need_reroute = true;
    for (auto edge_iter = netlist[i].edgelist.begin();
         edge_iter != netlist[i].edgelist.end(); edge_iter++) {
      if ((*edge_iter)->scale > 1) {
        num_reroute++;
        no_need_reroute = false;
        break;
      }
    }
    if (!no_need_reroute) {
      for (auto edge_iter = netlist[i].edgelist.begin();
           edge_iter != netlist[i].edgelist.end(); edge_iter++) {
        if ((*edge_iter)->scale > 1) {

        } else {
          (*edge_iter)->utilization -= (netlist[i].min_width + 1);
          (*edge_iter)->compute_cost();
        }
      }
      netlist[i].reroute = true;
      netlist[i].pathlist.clear();
      netlist[i].edgelist.clear();
      netlist[i].nodelist.clear();
    } else
      netlist[i].reroute = false;
  }
  cout << "num of nets to reroute: " << num_reroute << endl;
}

int main(int argc, char* argv[]) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
  /*if(argc != 2)
  {
      cout<<"usage: ./a.out [input]"<<endl;
      return 0;
  }*/
  ifstream infile(argv[1]);

  if (!infile.is_open()) {
    cout << "unable to open input file" << endl;
    return 0;
  }
  Info info;

  Graph graph;

  galois::preAlloc(numThreads * 2);
  nodeArray nodelist;
  edgeArray v_edge;
  edgeArray h_edge;
  Net* netlist;
  std::vector<GNode> node_GNode;
  const int MAX_ITERATION = 1;
  readfile(infile, info, nodelist, v_edge, h_edge, netlist, graph, node_GNode);
  infile.close();
  // cout<<"reading input file done"<<endl;
  timerstruct Time;
  // bool print = false;
  galois::InsertBag<int> allNodeBag;
  // galois::on_each(
  //        [&] (const unsigned tid, const unsigned numT)
  {
    unsigned total_nodes = (info.grid_x - 1) * (info.grid_y - 1);
    unsigned start       = 0;   // total_nodes / numT * tid;
    unsigned end = total_nodes; //(tid == numT - 1)? total_nodes : total_nodes /
                                // numT * (tid + 1);
    for (unsigned i = start; i < end; i++)
      allNodeBag.push(i);
  }
  //        );

  for (int i = 0; i < MAX_ITERATION; i++) {
    cout << endl;
    cout << "iteration: " << i << " K = " << K << " HEIGHT = " << HEIGHT
         << " slope = " << slope << endl;
    if (i != 0)
      reset_net(info, nodelist, v_edge, h_edge, netlist);

    Time.t_maze_route.start();

    galois::runtime::profileVtune(
        [&](void) {
          mazeroute(info, nodelist, allNodeBag, v_edge, h_edge, netlist, graph,
                    node_GNode, Time);
        },
        "mazeroute");
    Time.t_maze_route.stop();
    bool print = false; // Martin
    if (i == MAX_ITERATION - 1)
      print = true;
    int overflow =
        check_overflow(info, nodelist, v_edge, h_edge, netlist, print);
    if (overflow == 0)
      break;

    if (K < 1)
      K += K_STEP;
    else
      K = 2.0;
    HEIGHT += HEIGHT_STEP;
    if (i > 20)
      HEIGHT_STEP = 4;
  }

  Time.t_print_path.start();
  print_path(info, netlist);
  Time.t_print_path.stop();

  return 0;
}
