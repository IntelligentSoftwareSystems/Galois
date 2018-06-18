#ifndef VERTEX_H
#define VERTEX_H

#include "EquationSystem.h"
#include "NumaEquationSystem.h"

typedef enum { ROOT = 0, NODE, LEAF } VertexType;

class Vertex {
public:
  Vertex* left;
  Vertex* right;
  Vertex* parent;
  VertexType type;

  EquationSystem* system;

  Vertex(Vertex* Left, Vertex* Right, Vertex* Parent, VertexType type,
         int systemSize);
  Vertex(Vertex* Left, Vertex* Right, Vertex* Parent, VertexType type,
         int systemSize, int node);

  ~Vertex();

  void setLeft(Vertex* v);
  void setRight(Vertex* v);
  void setType(VertexType t);
};

#endif
