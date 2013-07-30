//Perform graph layout, Daniel Tunkelang style

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
const double Tmax = 255.00;
const double Tmin = 1;

const double gravConst = .0625;//.0625;
const int randRange = 1;
const double idealDist = 4;
const double lenLimit = 0.001;
const int RmaxMult = 16;
const bool useScaling = true;

bool doExit = false;
double alpha = 1.0;
double nnodes = 0.0;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<bool> opt_no_gravity("no-gravity", cll::desc("no gravity"), cll::Optional, cll::init(false));
static cll::opt<bool> opt_no_random("no-random", cll::desc("no random"), cll::Optional, cll::init(false));
static cll::opt<bool> opt_no_repulse("no-repulse", cll::desc("no repulse"), cll::Optional, cll::init(false));
static cll::opt<bool> opt_no_spring("no-spring", cll::desc("no spring"), cll::Optional, cll::init(false));
static cll::opt<bool> noLimit("no-limit", cll::desc("no limit to time steps"), cll::Optional, cll::init(false));


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
    if (!std::isfinite(x)) x = 0.0;
    if (!std::isfinite(y)) y = 0.0;
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
  Point<int> coord;
  double temp;
  double mass;

  Vertex()
    : position(randPoint() * idealDist * idealDist),
      temp(Tinit),
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
    
    // if (true) {
    //   for (Graph::iterator ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
    //     Vertex& v = g.getData(*ii);
    //     drawline(data, v.coord.x, v.coord.y, 
    //              v.coord.x + WIDTH * v.fGravity.x / W, v.coord.y + HEIGHT * v.fGravity.y / H,
    //              green);
    //     drawline(data, v.coord.x, v.coord.y, 
    //              v.coord.x + WIDTH * v.fSpring.x / W, v.coord.y + HEIGHT * v.fSpring.y / H,
    //              green2);
    //     drawline(data, v.coord.x, v.coord.y, 
    //              v.coord.x + WIDTH * v.fRepulse.x / W, v.coord.y + HEIGHT * v.fRepulse.y / H,
    //              green3);
    //     drawline(data, v.coord.x, v.coord.y, 
    //              v.coord.x + WIDTH * v.fRandom.x / W, v.coord.y + HEIGHT * v.fRandom.y / H,
    //              green4);
    //   }
    //   drawstring(data, 8, 8, "gravity", green);
    //   drawstring(data, 8, 20, "spring", green2);
    //   drawstring(data, 8, 32, "repulse", green3);
    //   drawstring(data, 8, 48, "random", green4);
    // }
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
    v.mass = 1 + std::distance(g.edge_begin(*ii), g.edge_end(*ii));
    v.position *= idealDist * idealDist * (1.0 / v.mass);
  }
}

//Computes spring force given quadradic spring with natural length of zero
Point<double> computeForceSpringZero(Vertex& v, GNode n, Graph& g) {
  Point<double> fSpring = {0.0,0.0};
  for(Graph::edge_iterator ee = g.edge_begin(n), eee = g.edge_end(n); ee!=eee; ee++) {
    Vertex& u2 = g.getData(g.getEdgeDst(ee));
    Point<double> change = u2.position - v.position;
    double lenSQ = change.lengthSQ();
    if (lenSQ < lenLimit) { //not stable
      fSpring += randPoint();
    } else {
      double len = sqrt(lenSQ);
      auto dir = change / len;
      double force = lenSQ * (1/idealDist);
      fSpring += dir * force;
    }
  }
  return fSpring;
}

Point<double> computeForceRepulsive(Vertex& v, GNode n, Graph& g) {
  Point<double> fRepulse = {0.0,0.0};
  for (Graph::iterator jj = g.begin(), ej = g.end(); jj!=ej; jj++) {
    if (n != *jj) {
      Vertex& u = g.getData(*jj, Galois::NONE);
      auto change = v.position - u.position;
      double lenSQ = change.lengthSQ();
      if (lenSQ < lenLimit) { // not stable
        fRepulse += randPoint();
      } else {
        double len = sqrt(lenSQ);
        auto dir = change / len;
        if (true) {
          if (len <= idealDist)
            fRepulse += dir * (idealDist * idealDist) / len; //short range
          else
            fRepulse += dir * (idealDist * idealDist * idealDist) / lenSQ; //long range
        } else {
          fRepulse += dir * alpha * (idealDist * idealDist) / len; //short range
          fRepulse += dir * (1.0 - alpha) * (idealDist * idealDist * idealDist) / lenSQ; //long range
        }
      }
    }
  }
  return fRepulse;
}

Point<double> computeForceRandom(Vertex& v, GNode n, Graph& g) {
  if ( drand48() < (v.temp / Tmax))
    return randPoint() * idealDist * Tmax / v.temp;
  else
    return Point<double>(); // randPoint() * v.temp / Tmax;
}

Point<double> computeForceGravity(Vertex& v, GNode n, Graph& g) {
  Point<double> dir = Point<double>(0,0) - v.position;
  return dir * gravConst;
}

static double maxGravity = 0.0;
static double maxRepulse = 0.0;
static double maxSpring = 0.0;

//takes time step, returns remainder of timestep
double updatePosition(Vertex& v, double time, Point<double> force) {
  double sf = useScaling ? (v.temp / Tmax) : 1.0;
  v.velocity = force * sf / v.mass;
  auto oldPos = v.position;
  double step = time;
  double len = sqrt(v.velocity.lengthSQ());
  while(sqrt(v.temp) < step * len)
    step /= 2;
  v.position += v.velocity * step;
  v.position.boundBy(Point<double>(-1000000,-1000000),Point<double>(1000000, 1000000));
  v.velocity = v.position - oldPos;
  return step;
}

void updateTemp(Vertex& v) {
  v.temp *= 0.99;//(1/nnodes);
  //v.temp = std::max(0.5, std::min(Tmax, idealDist * v.velocity.lengthSQ())); 
}

struct computeImpulse {
  Graph& g;
  computeImpulse(Graph& _g) :g(_g) {}

  template<typename Context>
  void operator()(GNode& n, Context& cnx) {
    Vertex& v = g.getData(n);
    doNode(n, v);
    if (v.temp > Tmin)
      cnx.push(n);
  }

  void operator()(GNode& n) {
    Vertex& v = g.getData(n);
    doNode(n, v);
  }

  void doNode(GNode& n, Vertex& v) {
    double time = 1.0;
    do {
      Point<double> fGravity, fRepulse, fSpring, fRandom;
      if (!opt_no_random)
        fRandom = computeForceRandom(v, n, g);
      if (!opt_no_gravity)
        fGravity = computeForceGravity(v, n, g);
      if (!opt_no_spring)
        fSpring = computeForceSpringZero(v,n,g);
      if (!opt_no_repulse)
        fRepulse = computeForceRepulsive(v, n, g);
      double fr = fRepulse.lengthSQ(), fs = fSpring.lengthSQ();
      double step = (fr > 4 * fs || fs > 4 * fr) ? time / 2 : time;
      time -= updatePosition(v, step, fGravity + fRepulse + fSpring + fRandom);

      if (false) {
        if (fGravity.lengthSQ() > maxGravity)
          maxGravity = fGravity.lengthSQ();
        if (fSpring.lengthSQ() > maxSpring)
          maxSpring = fSpring.lengthSQ();
        if (fRepulse.lengthSQ() > maxRepulse)
          maxRepulse = fRepulse.lengthSQ();
      }

    } while (time > 0.005);
    updateTemp(v);
  }
};

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
    //    std::cout << v.position.x << "," << v.position.y << " ";
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
    Tglob += v.temp;
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

  nnodes = std::distance(graph.begin(), graph.end());
  int Rmax = RmaxMult*nnodes;

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

  graph.getData(*graph.begin()).position = Point<double>(0,0);
  graph.getData(*graph.begin()).velocity = Point<double>(0,0);

  //Begin stages of algorithm
  Galois::StatTimer T;
  T.start();
  while(!doExit && (noLimit || ((Tglob > Tmin) && (numRounds < Rmax)))) {

    //Compute v's impulse
    Galois::do_all(graph.begin(), graph.end(), computeImpulse(graph));
    alpha *= .995;
    //Galois::for_each(graph.begin(), graph.end(), computeImpulse(graph));
    
    if (false && numRounds % 64 == 0) { //false && numRounds % 4 == 0) {
      computeCoord(graph);
      renderGraph(buffer, WIDTH, HEIGHT, graph, true);// SDL_Delay(1000/60))
      render(data_sf);
      init_data(buffer);
    }
    // char foo;
    // std::cin >> foo;


    //Update Global temperature
    computeGlobalTemp(graph, Tglob);

    std::cout << "Max ROUNDS: " << Rmax << " THIS ROUND: " << numRounds << " Tglobal " << Tglob << "\n";
    numRounds++;
  }
  T.stop();
  std::cout << "Exited loop.  Max Gravity " << sqrt(maxGravity) << " Max Spring " << sqrt(maxSpring) << " Max Repulse " << sqrt(maxRepulse) << "\n";
  computeCoord(graph);
  init_data(buffer);
  renderGraph(buffer, WIDTH, HEIGHT, graph, false);
  while (!doExit) {
    SDL_Delay(1000/30);
    render(data_sf);
  }
  return 0;
}
