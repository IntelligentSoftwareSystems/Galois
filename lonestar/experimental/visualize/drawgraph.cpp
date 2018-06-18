// Perform graph layout, Daniel Tunkelang style

#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include <iostream>
#include <stdlib.h>
#include "llvm/Support/CommandLine.h"
#include <cmath>
#include "Lonestar/BoilerPlate.h"
#include <SDL/SDL.h>
#include "drawing.h"

const double PI    = 3.141593;
const double Tinit = 255;
// const double Tmax = 255.00;
// const double Tmin = 1;

const double gravConst = .0625; //.0625;
// const int randRange = 1;
const double idealDist     = 4;
const double lenLimit      = 0.001;
const double maxForceLimit = 50;
const double totForceLimit = 500;
// const int RmaxMult = 16;
// const bool useScaling = false;

bool doExit     = false;
double alpha    = 1.0;
double nnodes   = 0.0;
double timestep = 0.0;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);
static cll::opt<bool> opt_no_gravity("no-gravity", cll::desc("no gravity"),
                                     cll::Optional, cll::init(false));
static cll::opt<bool> opt_no_repulse("no-repulse", cll::desc("no repulse"),
                                     cll::Optional, cll::init(false));
static cll::opt<bool> opt_no_spring("no-spring", cll::desc("no spring"),
                                    cll::Optional, cll::init(false));
static cll::opt<bool> noLimit("no-limit", cll::desc("no limit to time steps"),
                              cll::Optional, cll::init(false));
static cll::opt<double> natStepSize("step-size", cll::desc("Maximum step size"),
                                    cll::Optional, cll::init(1.0));
static cll::opt<int> printEvery("print",
                                cll::desc("Iterations between printing"),
                                cll::Optional, cll::init(1));

static bool init_app(const char* name, SDL_Surface* icon, uint32_t flags) {
  atexit(SDL_Quit);
  if (SDL_Init(flags) < 0)
    return 0;

  SDL_WM_SetCaption(name, name);
  SDL_WM_SetIcon(icon, NULL);

  return 1;
}

static void init_data(struct rgbData data[][WIDTH]) {
  memset(data, 255, WIDTH * HEIGHT * sizeof(rgbData));
}

static void render(SDL_Surface* sf) {
  SDL_Surface* screen = SDL_GetVideoSurface();
  if (SDL_BlitSurface(sf, NULL, screen, NULL) == 0)
    SDL_UpdateRect(screen, 0, 0, 0, 0);
}

static int filter(const SDL_Event* event) {
  if (event->type == SDL_QUIT)
    doExit = true;
  return event->type == SDL_QUIT;
}

template <typename T>
class Point {
public:
  T x, y;

  Point(T _x = 0, T _y = 0) : x(_x), y(_y) {}

  bool operator==(const Point& other) { return x == other.x && y == other.y; }

  Point operator*(T val) { return Point(x * val, y * val); }

  Point operator/(T val) { return Point(x / val, y / val); }

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
    if (!std::isfinite(x)) {
      abort();
      x = 0.0;
    }
    if (!std::isfinite(y)) {
      abort();
      y = 0.0;
    }
    x = std::max(x, cornerLow.x);
    y = std::max(y, cornerLow.y);
    x = std::min(x, cornerHigh.x);
    y = std::min(y, cornerHigh.y);
  }

  T lengthSQ() const { return x * x + y * y; }

  T vlog() const { return log(sqrt(lengthSQ())); }
};

Point<double> randPoint() {
  double angle = 2 * PI * drand48();
  double x     = cos(angle);
  double y     = sin(angle);
  return Point<double>(x, y);
}

template <typename T1, typename T2>
Point<T1> operator+(const Point<T1>& lhs, const Point<T2>& rhs) {
  return Point<T1>(lhs.x + rhs.x, lhs.y + rhs.y);
}
template <typename T1, typename T2>
Point<T1> operator-(const Point<T1>& lhs, const Point<T2>& rhs) {
  return Point<T1>(lhs.x - rhs.x, lhs.y - rhs.y);
}

class Vertex {
  Point<double> pos[2];

public:
  Point<double> force;
  Point<int> coord;
  double temp;
  double mass;

  Vertex() : temp(Tinit), mass(1.0) {}

  Point<double>& position(int which) { return pos[which]; }

  void dump() {
    std::cout << "V Pos [" << position(0).x << "," << position(0).y << " -- "
              << position(1).x << "," << position(1).y << "], ";
    std::cout << "F " << force.x << "," << force.y << " mass " << mass << "\n";
  }
};

typedef galois::graphs::LC_Linear_Graph<Vertex, unsigned int> Graph;
typedef Graph::GraphNode GNode;

double W = WIDTH, H = HEIGHT;
int midX, midY;

static bool renderGraph(struct rgbData data[][WIDTH], unsigned width,
                        unsigned height, Graph& g, int which, bool forces) {
  for (SDL_Event event; SDL_PollEvent(&event);)
    if (event.type == SDL_QUIT)
      return 0;

  rgbData black = {0, 0, 0};
  rgbData red   = {255, 0, 0};
  // rgbData green = {0,255,0};
  rgbData blue = {0, 0, 255};
  // rgbData green2 = {0,255,255};
  // rgbData green3 = {128,0,128};
  // rgbData green4 = {0,127,0};

  rgbData color = {200, 70, 110};
  drawcircle(data, midX, midY, 10, color);

  // clear
  // edges
  if (true)
    galois::do_all(g.begin(), g.end(), [&g, &data, black](GNode& n) {
      Vertex& v = g.getData(n);
      for (Graph::edge_iterator ee = g.edge_begin(n), eee = g.edge_end(n);
           ee != eee; ee++) {
        GNode n3  = g.getEdgeDst(ee);
        Vertex& u = g.getData(n3);
        drawline(data, v.coord.x, v.coord.y, u.coord.x, u.coord.y, black);
      }
    });
  if (forces) {
    if (true)
      galois::do_all(g.begin(), g.end(), [&g, &data, which, red](GNode& n) {
        Vertex& v     = g.getData(n);
        auto velocity = v.position(which) - v.position((which + 1) % 2);
        drawline(data, v.coord.x, v.coord.y, v.coord.x - WIDTH * velocity.x / W,
                 v.coord.y - HEIGHT * velocity.y / H, red);
      });

    // if (true) {
    //   for (Graph::iterator ii = g.begin(), ei = g.end(); ii!=ei; ii++) {
    //     Vertex& v = g.getData(*ii);
    //     drawline(data, v.coord.x, v.coord.y,
    //              v.coord.x + WIDTH * v.fGravity.x / W, v.coord.y + HEIGHT *
    //              v.fGravity.y / H, green);
    //     drawline(data, v.coord.x, v.coord.y,
    //              v.coord.x + WIDTH * v.fSpring.x / W, v.coord.y + HEIGHT *
    //              v.fSpring.y / H, green2);
    //     drawline(data, v.coord.x, v.coord.y,
    //              v.coord.x + WIDTH * v.fRepulse.x / W, v.coord.y + HEIGHT *
    //              v.fRepulse.y / H, green3);
    //     drawline(data, v.coord.x, v.coord.y,
    //              v.coord.x + WIDTH * v.fRandom.x / W, v.coord.y + HEIGHT *
    //              v.fRandom.y / H, green4);
    //   }
    //   drawstring(data, 8, 8, "gravity", green);
    //   drawstring(data, 8, 20, "spring", green2);
    //   drawstring(data, 8, 32, "repulse", green3);
    //   drawstring(data, 8, 48, "random", green4);
    // }
  }
  if (true)
    for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ii++) {
      Vertex& v = g.getData(*ii);
      drawcircle(data, v.coord.x, v.coord.y, 2, blue);
    }
  return 1;
}

void initializeVertices(Graph& g) {
  for (auto ii = g.begin(), ei = g.end(); ii != ei; ii++) {
    Vertex& v     = g.getData(*ii);
    v.mass        = 1 + std::distance(g.edge_begin(*ii), g.edge_end(*ii));
    v.position(0) = v.position(1) = randPoint() * nnodes / v.mass;
  }
}

// Computes spring force given quadradic spring with natural length of zero
Point<double> computeForceSpringZero(int which, Point<double> position, GNode n,
                                     Graph& g) {
  Point<double> fSpring = {0.0, 0.0};
  for (Graph::edge_iterator ee = g.edge_begin(n), eee = g.edge_end(n);
       ee != eee; ee++) {
    Vertex& u2           = g.getData(g.getEdgeDst(ee));
    Point<double> change = u2.position(which) - position;
    double lenSQ         = change.lengthSQ();
    if (lenSQ < lenLimit) { // not stable
      fSpring += randPoint();
    } else {
      double len   = sqrt(lenSQ);
      auto dir     = change / len;
      double force = lenSQ * (1 / idealDist);
      fSpring += dir * force;
    }
  }
  return fSpring;
}

Point<double> computeForceRepulsive(int which, Point<double> position, GNode n,
                                    Graph& g) {
  Point<double> fRepulse = {0.0, 0.0};
  for (Graph::iterator jj = g.begin(), ej = g.end(); jj != ej; jj++) {
    if (n != *jj) {
      Vertex& u    = g.getData(*jj, galois::MethodFlag::UNPROTECTED);
      auto change  = position - u.position(which);
      double lenSQ = change.lengthSQ();
      if (lenSQ < lenLimit) { // not stable
        fRepulse += randPoint();
      } else {
        double len = sqrt(lenSQ);
        auto dir   = change / len;
        if (true) {
          Point<double> localR;
          if (len < lenLimit) {
          } else if (len <= idealDist) {
            localR = dir * (idealDist * idealDist) / len; // short range
          } else {
            localR =
                dir * (idealDist * idealDist * idealDist) / lenSQ; // long range
          }
          fRepulse += localR;
        } else {
          fRepulse += dir * alpha * (idealDist * idealDist) / len; // short
                                                                   // range
          fRepulse += dir * (1.0 - alpha) *
                      (idealDist * idealDist * idealDist) / lenSQ; // long range
        }
      }
    }
  }
  return fRepulse;
}

Point<double> computeForceGravity(Point<double> position) {
  Point<double> dir = Point<double>(0, 0) - position;
  return dir * gravConst;
}

// takes time step, returns remainder of timestep
Point<double> updatePosition(Point<double> position, Vertex& v, double step,
                             Point<double> force) {
  auto velocity = force / v.mass;
  auto newPos   = position + velocity * step;
  newPos.boundBy(Point<double>(-1000000, -1000000),
                 Point<double>(1000000, 1000000));
  return newPos;
}

void updateTemp(Vertex& v) {
  v.temp *= 0.99; //(1/nnodes);
  // v.temp = std::max(0.5, std::min(Tmax, idealDist * v.velocity.lengthSQ()));
}

struct computeImpulse {
  Graph& g;
  int which;
  double stepSize;
  galois::substrate::PerThreadStorage<double>& maxForceSQ;
  galois::substrate::PerThreadStorage<Point<double>>& totalForce;

  computeImpulse(Graph& _g, int w, double ss,
                 galois::substrate::PerThreadStorage<double>& mf,
                 galois::substrate::PerThreadStorage<Point<double>>& tf)
      : g(_g), which(w), stepSize(ss), maxForceSQ(mf), totalForce(tf) {}

  void operator()(GNode& n) const {
    Vertex& v = g.getData(n);
    doNode(n, v);
  }

  void doNode(GNode& n, Vertex& v) const {
    Point<double> pos = v.position(which);
    Point<double> fGravity, fRepulse, fSpring;
    if (!opt_no_gravity)
      fGravity = computeForceGravity(pos);
    if (!opt_no_spring)
      fSpring = computeForceSpringZero(which, pos, n, g);
    if (!opt_no_repulse)
      fRepulse = computeForceRepulsive(which, pos, n, g);
    v.force                     = fGravity + fRepulse + fSpring;
    v.position((which + 1) % 2) = updatePosition(pos, v, stepSize, v.force);
    *totalForce.getLocal() += v.force;
    if (*maxForceSQ.getLocal() < v.force.lengthSQ())
      *maxForceSQ.getLocal() = v.force.lengthSQ();
    updateTemp(v);
  }
};

struct recomputePos {
  Graph& g;
  int which;
  double step;
  recomputePos(Graph& _g, int w, double s) : g(_g), which(w), step(s) {}

  void operator()(GNode& n) const {
    Vertex& v = g.getData(n);
    v.position((which + 1) % 2) =
        updatePosition(v.position(which), v, step, v.force);
    updateTemp(v);
  }
};

void computeCoord(int which, Graph& g) {
  Point<double> LB(1000000, 1000000), UB(-1000000, -1000000);
  for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ii++) {
    Vertex& v     = g.getData(*ii);
    auto position = v.position(which);
    if (position.x < LB.x)
      LB.x = position.x;
    if (position.x > UB.x)
      UB.x = position.x;
    if (position.y < LB.y)
      LB.y = position.y;
    if (position.y > UB.y)
      UB.y = position.y;
    //    std::cout << v.position.x << "," << v.position.y << " ";
  }
  double w = UB.x - LB.x;
  double h = UB.y - LB.y;
  for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ii++) {
    Vertex& v = g.getData(*ii);
    v.coord.x = (int)(WIDTH * (v.position(which).x - LB.x) / w);
    v.coord.y = (int)(WIDTH * (v.position(which).y - LB.y) / h);
  }
  std::cout << "Size: " << w << "x" << h << " given LB " << LB.x << "," << LB.y
            << " UB " << UB.x << "," << UB.y << "\n";
  W    = w;
  H    = h;
  midX = (int)(WIDTH * (0 - LB.x) / w);
  midY = (int)(WIDTH * (0 - LB.y) / h);
}

void computeGlobalTemp(Graph& g, double& Tglob) {
  int numNode = 0;
  Tglob       = 0.0;
  for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ii++) {
    ++numNode;
    Vertex& v = g.getData(*ii);
    Tglob += v.temp;
  }
  Tglob = Tglob / numNode;
}

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  Graph graph;

  // read in graph??
  galois::graphs::readGraph(graph, filename);
  // graph.structureFromFile(filename);

  nnodes = std::distance(graph.begin(), graph.end());

  // assign points in space to nodes
  initializeVertices(graph);

  // int Rmax = RmaxMult*nnodes;

  double Tglob = 0.0;
  computeGlobalTemp(graph, Tglob);

  int numRounds = 0;

  // Initialize visualization
  struct rgbData buffer[HEIGHT][WIDTH];
  // rgbData* buffer= new rgbData[WIDTH*HEIGHT];

  bool ok = init_app("SDL example", NULL, SDL_INIT_VIDEO) &&
            SDL_SetVideoMode(WIDTH, HEIGHT, 24, SDL_HWSURFACE);

  assert(ok);

  SDL_Surface* data_sf =
      SDL_CreateRGBSurfaceFrom((char*)buffer, WIDTH, HEIGHT, 24, WIDTH * 3,
                               0x000000FF, 0x0000FF00, 0x00FF0000, 0);

  SDL_SetEventFilter(filter);

  computeCoord(0, graph);
  computeCoord(1, graph);
  init_data(buffer);
  renderGraph(buffer, WIDTH, HEIGHT, graph, 0, false);
  render(data_sf);

  // Begin stages of algorithm
  galois::StatTimer T;
  T.start();

  double mf = 0.0;
  do {
    // Compute v's impulse
    galois::substrate::PerThreadStorage<double> maxForceSQ;
    galois::substrate::PerThreadStorage<Point<double>> totalForce;
    double step = natStepSize;
    // Vertex& v = graph.getData(*graph.begin());
    // v.dump();
    galois::do_all(
        graph.begin(), graph.end(),
        computeImpulse(graph, numRounds % 2, step, maxForceSQ, totalForce));
    // v.dump();
    Point<double> tf;
    mf = 0.0;
    for (int i = 0; i < maxForceSQ.size(); ++i) {
      if (mf < *maxForceSQ.getRemote(i))
        mf = *maxForceSQ.getRemote(i);
      tf += *totalForce.getRemote(i);
    }
    if (mf > maxForceLimit * maxForceLimit ||
        tf.lengthSQ() > totForceLimit * totForceLimit) {
      step = std::min((double)natStepSize,
                      std::min(maxForceLimit / sqrt(mf),
                               totForceLimit / sqrt(tf.lengthSQ())));
      galois::do_all(graph.begin(), graph.end(),
                     recomputePos(graph, numRounds % 2, step));
      // v.dump();
    }
    //    alpha *= .995;

    if (numRounds % printEvery == 0) {
      computeCoord(numRounds % 2, graph);
      init_data(buffer);
      renderGraph(buffer, WIDTH, HEIGHT, graph, numRounds % 2,
                  true); // SDL_Delay(1000/60))
      render(data_sf);

      std::cout << "Round: " << numRounds << " Tglobal " << Tglob << " Step "
                << step << " Max Force " << sqrt(mf) << " Effective Max Force "
                << sqrt(mf) * step << " Total Force " << sqrt(tf.lengthSQ())
                << " Effective Total Force " << sqrt(tf.lengthSQ()) * step
                << " Average Force " << sqrt(tf.lengthSQ()) / nnodes
                << " Effective Average Force "
                << sqrt(tf.lengthSQ()) * step / nnodes << "\n";
    }
    // Update Global temperature
    // computeGlobalTemp(graph, Tglob);

    numRounds++;
  } while (!doExit && (noLimit || (mf > 0.1))); // && numRounds < Rmax)));
  T.stop();
  std::cout << "Exited loop.\n";
  computeCoord(numRounds % 2, graph);
  init_data(buffer);
  renderGraph(buffer, WIDTH, HEIGHT, graph, numRounds % 2, false);
  while (!doExit) {
    SDL_Delay(1000 / 30);
    render(data_sf);
  }
  return 0;
}
