#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/Graph/LCGraph.h"
#include <iostream>
#include <stdlib.h>
#include "llvm/Support/CommandLine.h"
#include <cmath>
#include "Lonestar/BoilerPlate.h"
#include <SDL/SDL.h>
#include "drawing.h"

const double PI = 3.141593;
const double Tinit = 255;
const double Tmax = 256.00;
const double Tmin = 3;

const double repulseK = 10;
const double gravConst = .0625;//.0625;
const double springK = .1;
const int randRange = 1;
const double idealDist = 8;
const double lenLimit = 0.001;
const int RmaxMult = 16;
const bool useMomentum = false;
const bool useScaling = true;
const bool noLimit = true;

bool doExit = false;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<bool> opt_no_gravity("no-gravity", cll::desc("no gravity"), cll::Optional, cll::init(false));
static cll::opt<bool> opt_no_random("no-random", cll::desc("no random"), cll::Optional, cll::init(false));
static cll::opt<bool> opt_no_repulse("no-repulse", cll::desc("no repulse"), cll::Optional, cll::init(false));
static cll::opt<bool> opt_no_spring("no-spring", cll::desc("no spring"), cll::Optional, cll::init(false));


static bool init_app(const char * name, SDL_Surface * icon, uint32_t flags)
{
  atexit(SDL_Quit);
  if(SDL_Init(flags) < 0)
    return 0;

  SDL_WM_SetCaption(name, name);
  SDL_WM_SetIcon(icon, NULL);

  return 1;
}

static void init_data(struct rgbData data[][WIDTH])
{
  memset(data, 255, WIDTH*HEIGHT*sizeof(rgbData));
}

static void render(SDL_Surface * sf)
{
  SDL_Surface * screen = SDL_GetVideoSurface();
  if(SDL_BlitSurface(sf, NULL, screen, NULL) == 0)
    SDL_UpdateRect(screen, 0, 0, 0, 0);
}

static int filter(const SDL_Event * event)
{ 
  if (event->type == SDL_QUIT)
    doExit = true;
  return event->type == SDL_QUIT;
}

template<typename T>
class Point {
public:
  T x, y;

  Point(T _x = 0, T _y = 0) :x(_x), y(_y) {}

  bool operator==(const Point& other) {
    return x == other.x && y == other.y;
  }

  Point operator*(T val) {
    return Point(x*val, y*val);
  }

  Point operator/(T val) {
    return Point(x/val, y/val);
  }

  Point& operator+=(const Point& other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  Point& operator*=(const Point& other) {
    x *= other.x;
    y *= other.y;
    return *this;
  }

  void boundBy(const Point& cornerLow, const Point& cornerHigh) {
    x = std::max(x, cornerLow.x);
    y = std::max(y, cornerLow.y);
    x = std::min(x, cornerHigh.x);
    y = std::min(y, cornerHigh.y);
  }

  T lengthSQ() const {
    return x * x + y * y;
  }

  T vlog() const {
    return log(sqrt(lengthSQ()));
  }
};

Point<double> randPoint() {
  double angle = 2 * PI * drand48();
  double x = cos(angle);
  double y = sin(angle);
  return Point<double>(x, y);
}

template<typename T1, typename T2>
Point<T1> operator+(const Point<T1>& lhs, const Point<T2>& rhs) {
  return Point<T1>(lhs.x + rhs.x, lhs.y + rhs.y);
}
template<typename T1, typename T2>
Point<T1> operator-(const Point<T1>& lhs, const Point<T2>& rhs) {
  return Point<T1>(lhs.x - rhs.x, lhs.y - rhs.y);
}

class Vertex {
public:
  Point<double> position;
  Point<double> velocity;
  Point<double> fGravity;
  Point<double> fRepulse;
  Point<double> fSpring;
  Point<double> fRandom;
  Point<int> coord;
  double t;
  double mass;

  Vertex()
    : position(randPoint() * idealDist),
      velocity(randPoint()*10),
      t(Tinit),
      mass(1.0)
  {}
};

typedef Galois::Graph::LC_Linear_Graph<Vertex, unsigned int> Graph;
typedef Graph::GraphNode GNode;


double W = WIDTH, H = HEIGHT;
int midX, midY;

static bool renderGraph(struct rgbData data[][WIDTH], unsigned width, unsigned height, Graph& g, bool forces)
{
  for(SDL_Event event; SDL_PollEvent(&event);)
    if(event.type == SDL_QUIT) return 0;

  rgbData black = {0,0,0};
  rgbData red = {255,0,0};
  rgbData green = {0,255,0};
  rgbData blue = {0,0,255};
  rgbData green2 = {0,255,255};
  rgbData green3 = {128,0,128};
  rgbData green4 = {0,127,0};

  drawcircle(data, midX, midY, 10, {200,70,110});

  //clear
  //edges
  if (true)
    Galois::do_all(g.begin(), g.end(), [&g, &data, black] (GNode& n) {
        Vertex& v = g.getData(n);
        for(Graph::edge_iterator ee = g.edge_begin(n), eee = g.edge_end(n); ee!=eee; ee++) {
          GNode n3 = g.getEdgeDst(ee);
          Vertex& u = g.getData(n3);
          drawline(data, v.coord.x, v.coord.y, u.coord.x, u.coord.y, black);
        }
      } );
  if (forces) {
    if (true)
      Galois::do_all(g.begin(), g.end(), [&g, &data, red] (GNode& n) {
        Vertex& v = g.getData(n);
        drawline(data, v.coord.x, v.coord.y, 
                 v.coord.x - WIDTH * v.velocity.x / W, v.coord.y - HEIGHT * v.velocity.y / H,
                 red);
        } );
    
    if (true) {
      for (Graph::iterator ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
        Vertex& v = g.getData(*ii);
        drawline(data, v.coord.x, v.coord.y, 
                 v.coord.x + WIDTH * v.fGravity.x / W, v.coord.y + HEIGHT * v.fGravity.y / H,
                 green);
        drawline(data, v.coord.x, v.coord.y, 
                 v.coord.x + WIDTH * v.fSpring.x / W, v.coord.y + HEIGHT * v.fSpring.y / H,
                 green2);
        drawline(data, v.coord.x, v.coord.y, 
                 v.coord.x + WIDTH * v.fRepulse.x / W, v.coord.y + HEIGHT * v.fRepulse.y / H,
                 green3);
        drawline(data, v.coord.x, v.coord.y, 
                 v.coord.x + WIDTH * v.fRandom.x / W, v.coord.y + HEIGHT * v.fRandom.y / H,
                 green4);
      }
      drawstring(data, 8, 8, "gravity", green);
      drawstring(data, 8, 20, "spring", green2);
      drawstring(data, 8, 32, "repulse", green3);
      drawstring(data, 8, 48, "random", green4);
    }
  }
  if (true)
    for (Graph::iterator ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
      Vertex& v = g.getData(*ii);
      drawcircle(data, v.coord.x, v.coord.y, 2, blue);
    }
  return 1;
}

void initializeVertices(Graph& g) {
  for (auto ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
    Vertex& v = g.getData(*ii);
    v.mass = 1 + std::distance(g.edge_begin(*ii), g.edge_end(*ii)) / 2;
    //    v.position *= 1 / v.mass;
  }
}

void computeBaryCen(Graph& g, double& cx, double& cy) {
  //Compute barycenter
  cx = 0;
  cy = 0;
  for(Graph::iterator ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
    GNode n = *ii;
    Vertex& v = g.getData(n);
    cx += v.position.x;
    cy += v.position.y;
  }
  cx /= std::distance(g.begin(), g.end());
  cy /= std::distance(g.begin(), g.end());
  //std::cout << " cx " << cx << " cy " << cy << "\n";
}

//Computes spring force given ideal spring with natural length
void computeForceSpringIdeal (GNode n, Graph& g) {
  Vertex& v = g.getData(n);
  v.fSpring = {0.0,0.0};
  for(Graph::edge_iterator ee = g.edge_begin(n), eee = g.edge_end(n); ee!=eee; ee++) {
    Vertex& u2 = g.getData(g.getEdgeDst(ee));
    Point<double> change = u2.position - v.position;
    double lenSQ = change.lengthSQ();
    if (lenSQ > lenLimit) {
      double len = sqrt(lenSQ);
      auto dir = change / len;
      double spring = len - idealDist;
      // if (idealDist < len)
      //   spring *= log(log(spring));
      double force = spring * springK;
      v.fSpring += dir * force;
    }
  }
}

//Computes spring force given ideal spring with natural length of zero
void computeForceSpringZero(GNode n, Graph& g) {
  Vertex& v = g.getData(n);
  v.fSpring = {0.0,0.0};
  for(Graph::edge_iterator ee = g.edge_begin(n), eee = g.edge_end(n); ee!=eee; ee++) {
    Vertex& u2 = g.getData(g.getEdgeDst(ee));
    Point<double> change = u2.position - v.position;
    double lenSQ = change.lengthSQ();
    if (lenSQ > lenLimit) {
      double len = sqrt(lenSQ);
      auto dir = change / len;
      double force = len * springK;
      v.fSpring += dir * force;
    }
  }
}

void computeForceRepulsive(GNode n, Graph& g) {
  Vertex& v = g.getData(n);
  v.fRepulse = {0.0,0.0};
  for (Graph::iterator jj = g.begin(), ej = g.end(); jj!=ej; jj++) {
    if (n != *jj) {
      Vertex& u = g.getData(*jj);
      auto change = v.position - u.position;
      double lenSQ = change.lengthSQ();
      auto dir = (lenSQ > lenLimit) ? change / sqrt(lenSQ) : randPoint();
      if (lenSQ > lenLimit)
        v.fRepulse += dir * repulseK * v.mass / lenSQ;
    }
  }
}

void computeForceRandom(GNode n, Graph& g) {
  Vertex& v = g.getData(n);
  v.fRandom = randPoint() * sqrt(sqrt(v.velocity.lengthSQ()));
}

void computeForceGravity(GNode n, Graph& g) {
  Vertex& v = g.getData(n);
  Point<double> dir = Point<double>(0,0) - v.position;
  v.fGravity = dir * gravConst;
}

void computeImpulse(Graph& g) {
  Galois::do_all(g.begin(), g.end(), [&g] (GNode& n) { 
      if (!opt_no_random)
        computeForceRandom(n, g);
      if (!opt_no_gravity)
        computeForceGravity(n, g);
      if (!opt_no_spring)
        computeForceSpringZero(n, g); 
      if (!opt_no_repulse)
        computeForceRepulsive(n, g);
    });
}

void updateVSimple(Graph& g) {
  for(Graph::iterator ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
    Vertex& v = g.getData(*ii);

    Point<double> pCurr = v.fGravity + v.fRepulse + v.fSpring + v.fRandom;
    double sf = useScaling ? ((double)v.t / Tmax) : 1.0;
    if (useMomentum)
      v.velocity = v.velocity + pCurr * sf / v.mass;
    else
      v.velocity = pCurr * sf / v.mass;
    auto oldPos = v.position;
    v.position += v.velocity;
    v.position.boundBy(Point<double>(-1000000,-1000000),Point<double>(1000000, 1000000));
    v.velocity = v.position - oldPos;
    //    v.t -= 1;
  }
}

void computeCoord(Graph& g) {
  Point<double> LB(1000000,1000000), UB(-1000000,-1000000);
  for(Graph::iterator ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
    Vertex& v = g.getData(*ii);
    if (v.position.x < LB.x)
      LB.x = v.position.x;
    if (v.position.x > UB.x)
      UB.x = v.position.x;
    if (v.position.y < LB.y)
      LB.y = v.position.y;
    if (v.position.y > UB.y)
      UB.y = v.position.y;
  }
  double w = UB.x - LB.x;
  double h = UB.y - LB.y;
  for(Graph::iterator ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
    Vertex& v = g.getData(*ii);
    v.coord.x = (int)(WIDTH * (v.position.x - LB.x) / w);
    v.coord.y = (int)(WIDTH * (v.position.y - LB.y) / h);
  }
  std::cout << "Size: " << w << "x" << h << " given LB " << LB.x << "," << LB.y << " UB " << UB.x << "," << UB.y << "\n";
  W = w; H = h;
  midX = (int)(WIDTH * (0 - LB.x) / w);
  midY = (int)(WIDTH * (0 - LB.y) / h);
}

void computeGlobalTemp(Graph& g, double& Tglob) {
  int numNode = 0;
  Tglob = 0.0;
  for (Graph::iterator ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
    ++numNode;
    Vertex& v = g.getData(*ii);
    Tglob += v.t;
  }
  Tglob = Tglob / numNode;
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  Graph graph;

  //read in graph??
  Galois::Graph::readGraph(graph, filename);
  //graph.structureFromFile(filename);

  //assign points in space to nodes
  initializeVertices(graph);

  int Rmax = RmaxMult*std::distance(graph.begin(), graph.end());

  double Tglob = 0.0;
  computeGlobalTemp(graph, Tglob);

  int numRounds = 0;

  //Initialize visualization
  struct rgbData buffer[HEIGHT][WIDTH];
  //rgbData* buffer= new rgbData[WIDTH*HEIGHT];

  bool ok =
    init_app("SDL example", NULL, SDL_INIT_VIDEO) &&
    SDL_SetVideoMode(WIDTH, HEIGHT, 24, SDL_HWSURFACE);

  assert(ok);

  init_data(buffer);

  SDL_Surface * data_sf = 
    SDL_CreateRGBSurfaceFrom((char*)buffer, WIDTH, HEIGHT, 24, WIDTH * 3,
                             0x000000FF, 0x0000FF00, 0x00FF0000, 0);
  
  SDL_SetEventFilter(filter);

  computeCoord(graph);
  renderGraph(buffer, WIDTH, HEIGHT, graph, false);
  render(data_sf);

  //Begin stages of algorithm
  Galois::StatTimer T;
  T.start();
  while(!doExit && (noLimit || (Tglob > Tmin) && (numRounds < Rmax))) {

    double cx = 0, cy = 0;
    computeBaryCen(graph,cx,cy);

    //Compute v's impulse
    computeImpulse(graph);

    if (numRounds % 10 == 0) { //false && numRounds % 4 == 0) {
      computeCoord(graph);
      renderGraph(buffer, WIDTH, HEIGHT, graph, true);// SDL_Delay(1000/60))
      render(data_sf);
      init_data(buffer);
    }
    // char foo;
    // std::cin >> foo;

    //Update v's Position and Temperature
    updateVSimple(graph);

    //Update Global temperature
    computeGlobalTemp(graph, Tglob);

    std::cout << "Max ROUNDS: " << Rmax << " THIS ROUND: " << numRounds << " Tglobal " << Tglob << "\n";
    numRounds++;
  }
  T.stop();
  std::cout << "Exited loop\n";
  computeCoord(graph);
  init_data(buffer);
  renderGraph(buffer, WIDTH, HEIGHT, graph, false);
  while (!doExit) {
    SDL_Delay(1000/30);
    render(data_sf);
  }
  return 0;
}
