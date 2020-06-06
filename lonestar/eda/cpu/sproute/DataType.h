#ifndef _DATATYPE_H_
#define _DATATYPE_H_

#define MAXDEMAND 500  // MAX # Segments over an edge
#define MAXLAYER 20    // MAX # Layer of a routing
#define FILESTRLEN 100 // MAX length of file name
#define BIG_INT 1e7    // big integer used as infinity

#define TRUE 1
#define FALSE 0
typedef char Bool;

typedef struct {
  Bool xFirst; // route x-direction first (only for L route)
  Bool HVH;    // TRUE = HVH or FALSE = VHV (only for Z route)
  Bool maze;   // Whether this segment is routed by maze

  short x1, y1, x2, y2; // coordinates of two endpoints
  int netID;            // the netID of the net this segment belonging to
  short Zpoint;         // The coordinates of Z point (x for HVH and y for VHV)
  short* route;         // array of H and V Edges to implement this Segment
  int numEdges;         // number of H and V Edges to implement this Segment
} Segment;              // A Segment is a 2-pin connection

typedef struct {
  char name[18]; // net name
  int netIDorg;  // orginal net ID in the input file
  short numPins; // number of pins in the net
  short deg; // net degree (number of MazePoints connecting by the net, pins in
             // same MazePoints count only 1)
  short* pinX; // array of X coordinates of pins
  short* pinY; // array of Y coordinates of pins
  short* pinL; // array of L coordinates of pins
  short minwidth;
} Net; // A Net is a set of connected MazePoints

typedef struct edge_t : public galois::runtime::Lockable {
  // galois::substrate::SimpleLock lock;
  int congCNT;
  int cap;                // the capacity of the edge
  std::atomic<int> usage; // the usage of the edge
  int red;
  int last_usage;
  float est_usage; // the estimated usage of the edge

  std::atomic<int> max_ripups;
  int max_have_rippedups;
  bool ripups_cur_round;
  /*inline void acquireLock()
  {
      lock.lock();
  }
  inline void releaseLock()
  {
      lock.unlock();
  } */
} Edge; // An Edge is the routing track holder between two adjacent MazePoints

typedef struct {
  unsigned short cap;   // the capacity of the edge
  unsigned short usage; // the usage of the edge
  unsigned short red;

} Edge3D;

typedef struct {
  Bool assigned;

  short status;
  short conCNT;
  short botL, topL;
  short heights[6];

  short x, y;    // position in the grid graph
  short nbr[3];  // three neighbors
  short edge[3]; // three adjacent edges
  int hID;
  int lID;
  int eID[6];
  int stackAlias;

} TreeNode;

#define NOROUTE 0
#define LROUTE 1
#define ZROUTE 2
#define MAZEROUTE 3

typedef char RouteType;

typedef struct {
  RouteType type; // type of route: LROUTE, ZROUTE, MAZEROUTE
  Bool
      xFirst; // valid for LROUTE, TRUE - the route is horizontal first (x1, y1)
              // - (x2, y1) - (x2, y2), FALSE (x1, y1) - (x1, y2) - (x2, y2)
  Bool
      HVH; // valid for ZROUTE, TRUE - the route is HVH shape, FALSE - VHV shape
  short Zpoint; // valid for ZROUTE, the position of turn point for Z-shape
  short*
      gridsX; // valid for MAZEROUTE, a list of grids (n=routelen+1) the route
              // passes, (x1, y1) is the first one, but (x2, y2) is the lastone
  short*
      gridsY; // valid for MAZEROUTE, a list of grids (n=routelen+1) the route
              // passes, (x1, y1) is the first one, but (x2, y2) is the lastone
  short* gridsL; // n
  int routelen;  // valid for MAZEROUTE, the number of edges in the route
  // Edge3D *edge;       // list of 3D edges the route go through;

} Route;

typedef struct {
  Bool assigned;

  int len; // the Manhanttan Distance for two end nodes
  int n1, n1a;
  int n2, n2a;
  Route route;

  int n_ripups;
  bool ripup;
} TreeEdge;

typedef struct {
  int deg;
  TreeNode* nodes; // the nodes (pin and Steiner nodes) in the tree
  TreeEdge* edges; // the tree edges
} StTree;

typedef struct {

  int treeIndex;
  int minX;
  float npv; // net length over pin
} OrderNetPin;

typedef struct {
  int length;
  int treeIndex;
  int xmin;
} OrderTree;

typedef struct {
  short l;
  int x, y;
} parent3D;

typedef struct {
  int length;
  int edgeID;
} OrderNetEdge;

typedef enum { NORTH, EAST, SOUTH, WEST, ORIGIN, UP, DOWN } dirctionT;
typedef enum { NONE, HORI, VERT, BID } viaST;

#endif /* _DATATYPE_H_ */
