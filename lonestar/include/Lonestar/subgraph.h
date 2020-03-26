#ifndef SUBGRAPH_H_
#define SUBGRAPH_H_

class Subgraph {
public:
  unsigned* n;         // n[l]: number of vertices in G_l
  unsigned** d;        // d[l]: degrees of G_l
  unsigned* adj;       // truncated list of neighbors
  unsigned char* lab;  // lab[i] label of vertex i
  unsigned** vertices; // sub[l]: vertices in G_l
  int core;
  unsigned max_size;
  Subgraph() {}
  Subgraph(unsigned c, unsigned k) { allocate(c, k); }
  ~Subgraph() {
    for (unsigned i = 2; i < max_size; i++) {
      if (d[i])
        free(d[i]);
      if (vertices[i])
        free(vertices[i]);
    }
    if (n)
      free(n);
    if (d)
      free(d);
    if (lab)
      free(lab);
    if (adj)
      free(adj);
    if (vertices)
      free(vertices);
  }
  void allocate(int c, unsigned k) {
    max_size = k;
    core     = c;
    n        = (unsigned*)calloc(k, sizeof(unsigned));
    d        = (unsigned**)malloc(k * sizeof(unsigned*));
    vertices = (unsigned**)malloc(k * sizeof(unsigned*));
    for (unsigned i = 2; i < k; i++) {
      d[i]        = (unsigned*)malloc(core * sizeof(unsigned));
      vertices[i] = (unsigned*)malloc(core * sizeof(unsigned));
    }
    lab = (unsigned char*)calloc(core, sizeof(unsigned char));
    adj = (unsigned*)malloc(core * core * sizeof(unsigned));
  }
};
#endif
