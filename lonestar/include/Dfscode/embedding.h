#ifndef __GRAPH_TYPES_HPP__
#define __GRAPH_TYPES_HPP__
#include "types.hpp"
#include "cgraph.hpp"
#include "dfs_code.hpp"

struct LocalStatus {
  int thread_id;
  unsigned task_split_level;
  unsigned embeddings_regeneration_level;
  unsigned current_dfs_level;
  int frequent_patterns_count;
  bool is_running;
  DFSCode DFS_CODE;
  DFSCode DFS_CODE_IS_MIN;
  CGraph GRAPH_IS_MIN;
  std::vector<std::deque<DFS>> dfs_task_queue;
  std::deque<DFSCode> dfscodes_to_process;
};
class EmbVector;

// An embedding consists of an edge (pointer)
// and an embedding pointer to its parent embedding
struct Embedding {
  unsigned num_vertices;
  LabEdge* edge;
  Embedding* prev;
  Embedding() : num_vertices(0), edge(0), prev(0){};
  std::string to_string() const {
    std::stringstream ss;
    ss << "[" << edge->to_string() << "]";
    // ss << "[" << num_vertices << ", " << edge->to_string() << "]";
    return ss.str();
  }
  std::string to_string_all() {
    std::vector<LabEdge> ev;
    ev.push_back(*edge);
    for (Embedding* p = prev; p; p = p->prev) {
      ev.push_back(*(p->edge));
    }
    std::reverse(ev.begin(), ev.end());
    std::stringstream ss;
    for (size_t i = 0; i < ev.size(); i++) {
      ss << ev[i].to_string() << "; ";
    }
    return ss.str();
  }
};

// Embedding list
class EmbeddingList : public std::vector<Embedding> {
public:
  void push(int n, LabEdge* edge, Embedding* prev) {
    Embedding d;
    d.num_vertices = n;
    d.edge         = edge;
    d.prev         = prev;
    push_back(d);
  }
  std::string to_string() const {
    std::stringstream ss;
    for (size_t i = 0; i < size(); i++)
      ss << (*this)[i].to_string() << "; ";
    return ss.str();
  }
};

// Stores information of edges/nodes that were already visited in the
// current DFS branch of the search.
class History : public std::vector<LabEdge*> {
private:
  std::set<int> edge;
  std::set<int> vertex;

public:
  bool hasEdge(unsigned id) { return (bool)edge.count(id); }
  bool hasEdge(LabEdge e) {
    for (std::vector<LabEdge*>::iterator it = this->begin(); it != this->end();
         ++it) {
      if ((*it)->from == e.from && (*it)->to == e.to &&
          (*it)->elabel == e.elabel)
        return true;
      else if ((*it)->from == e.to && (*it)->to == e.from &&
               (*it)->elabel == e.elabel)
        return true;
    }
    return false;
  }
  bool hasVertex(unsigned id) { return (bool)vertex.count(id); }
  History() {}
  History(Embedding* p) { build(p); }
  void build(Embedding* e) {
    if (e) {
      push_back(e->edge);
      edge.insert(e->edge->id);
      vertex.insert(e->edge->from);
      vertex.insert(e->edge->to);
      for (Embedding* p = e->prev; p; p = p->prev) {
        push_back(p->edge); // this line eats 8% of overall instructions(!)
        edge.insert(p->edge->id);
        vertex.insert(p->edge->from);
        vertex.insert(p->edge->to);
      }
      std::reverse(begin(), end());
    }
  }
  std::string to_string() const {
    std::stringstream ss;
    for (size_t i = 0; i < size(); i++) {
      ss << at(i)->to_string() << "; ";
    }
    return ss.str();
  }
};

typedef std::map<unsigned, std::map<unsigned, unsigned>> Map2D;
typedef std::map<int, std::map<int, std::map<int, EmbeddingList>>> PatternMap3D;
typedef std::map<int, std::map<int, EmbeddingList>> PatternMap2D;
typedef std::map<int, EmbeddingList> PatternMap1D;
typedef PatternMap3D::iterator EmbeddingList_iterator3;
typedef PatternMap2D::iterator EmbeddingList_iterator2;
typedef PatternMap1D::iterator EmbeddingList_iterator1;
typedef PatternMap3D::reverse_iterator EmbeddingList_riterator3;
// typedef std::vector<int> graph_id_list_t;
// typedef std::map<int, graph_id_list_t>   edge_gid_list1_t;
// typedef std::map<int, edge_gid_list1_t>  edge_gid_list2_t;
// typedef std::map<int, edge_gid_list2_t>  edge_gid_list3_t;

bool get_forward_rmpath(Graph& graph, std::vector<LabEdge>& edge_list,
                        LabEdge* e, LabelT minlabel, History& history,
                        EdgeList& result) {
  result.clear();
  assert(e->to >= 0 && e->to < graph.size());
  assert(e->from >= 0 && e->from < graph.size());
  LabelT tolabel = graph.getData(e->to);
  // Graph::edge_iterator first = graph.edge_begin(e->from,
  // galois::MethodFlag::UNPROTECTED); Graph::edge_iterator last =
  // graph.edge_end(e->from, galois::MethodFlag::UNPROTECTED); for (auto it =
  // first; it != last; ++ it) {
  for (auto it : graph.edges(e->from)) {
    GNode dst       = graph.getEdgeDst(it);
    auto elabel     = graph.getEdgeData(it);
    auto& dst_label = graph.getData(dst);
    if (e->to == dst || minlabel > dst_label || history.hasVertex(dst))
      continue;
    if (e->elabel < elabel || (e->elabel == elabel && tolabel <= dst_label)) {
      LabEdge* eptr = &(edge_list[*it]);
      result.push_back(eptr);
    }
  }
  return (!result.empty());
}

// e (from, elabel, to)
// this function takes a "pure" forward edge, that is: an edge that
// extends the last node of the right-most path, i.e., the right-most node.
bool get_forward_pure(Graph& graph, std::vector<LabEdge>& edge_list, LabEdge* e,
                      LabelT minlabel, History& history, EdgeList& result) {
  result.clear();
  assert(e->to >= 0 && e->to < graph.size());
  // Walk all edges leaving from vertex e->to.
  // Graph::edge_iterator first = graph.edge_begin(e->to);
  // Graph::edge_iterator last = graph.edge_end(e->to);
  // for (auto it = first; it != last; ++ it) {
  for (auto it : graph.edges(e->to)) {
    GNode dst = graph.getEdgeDst(it);
    assert(dst >= 0 && dst < graph.size());
    // auto elabel = graph.getEdgeData(it);
    auto& dst_label = graph.getData(dst);
    if (minlabel > dst_label || history.hasVertex(dst))
      continue;
    LabEdge* eptr = &(edge_list[*it]);
    result.push_back(eptr);
  }
  return (!result.empty());
}
/*
bool get_forward_root(Graph &graph, std::vector<LabEdge>& edge_list, const
VeridT src, EdgeList &result) { result.clear(); auto& src_label =
graph.getData(src);
    //Graph::edge_iterator first = graph.edge_begin(src,
galois::MethodFlag::UNPROTECTED);
    //Graph::edge_iterator last = graph.edge_end(src,
galois::MethodFlag::UNPROTECTED);
    //for (auto it = first; it != last; ++ it) {
    for (auto it : graph.edges(src)) {
        GNode dst = graph.getEdgeDst(it);
        assert(dst >= 0 && dst < graph.size());
        //auto elabel = graph.getEdgeData(it);
        auto& dst_label = graph.getData(dst);
        if(src_label <= dst_label) {
            LabEdge * eptr = &(edge_list[*it]);
            //std::cout << eptr->to_string() << " ";
            result.push_back(eptr);
        }
    }
    //std::cout << "\n";
    return (!result.empty());
}
//*/

// get_backward (graph, e1, e2, history);
//   e1 (from1, elabel1, to1)
//   e2 (from2, elabel2, to2)
//   to2 -> from1
//   (elabel1 < elabel2 || (elabel == elabel2 && tolabel1 < tolabel2) .
//   (elabel1, to1)
// RK comment: gets backward edge that starts and ends at the right most path
// e1 is the forward edge and the backward edge goes to e1->from
LabEdge* get_backward(Graph& graph, std::vector<LabEdge>& edge_list,
                      LabEdge* e1, LabEdge* e2, History& history) {
  if (e1 == e2)
    return 0;
  assert(e1->from >= 0 && e1->from < graph.size());
  assert(e1->to >= 0 && e1->to < graph.size());
  assert(e2->to >= 0 && e2->to < graph.size());
  VeridT src = e2->to;
  // Graph::edge_iterator first = graph.edge_begin(src);
  // Graph::edge_iterator last = graph.edge_end(src);
  // for (auto it = first; it != last; ++ it) {
  for (auto it : graph.edges(src)) {
    GNode dst   = graph.getEdgeDst(it);
    auto elabel = graph.getEdgeData(it);
    // if(history.hasEdge(LabEdge(src, dst, elabel, *it))) continue;
    if (history.hasEdge(*it))
      continue;
    if ((dst == e1->from) &&
        ((e1->elabel < elabel) ||
         ((e1->elabel == elabel) &&
          (graph.getData(e1->to) <= graph.getData(e2->to))))) {
      return &(edge_list[*it]);
    }
  }
  return 0;
}

////////////////////////////////////////////////
// Funstions used for regenerating embeddings //
////////////////////////////////////////////////
bool get_forward(Graph& graph, std::vector<LabEdge>& edge_list,
                 const DFSCode& DFS_CODE, History& history, EdgeList& result) {
  result.clear();
  // forward extenstion from dfs_from <=> from
  VeridT dfs_from = DFS_CODE.back().from;
  VeridT from     = (VeridT)-1;
  // skip the last one in dfs code
  // get the "from" vertex id from the history
  for (size_t i = DFS_CODE.size() - 2; i >= 0; i--) {
    if (dfs_from == DFS_CODE[i].from) {
      from = history[i]->from;
      break;
    }
    if (dfs_from == DFS_CODE[i].to) {
      from = history[i]->to;
      break;
    }
  }
  assert(from != (VeridT)-1);
  DFS dfs = DFS_CODE.back();
  for (auto it : graph.edges(from)) {
    GNode dst   = graph.getEdgeDst(it);
    auto elabel = graph.getEdgeData(it);
    if (elabel == dfs.elabel && graph.getData(dst) == dfs.tolabel &&
        !history.hasVertex(dst))
      result.push_back(&(edge_list[*it]));
  }
  return (!result.empty());
}

LabEdge* get_backward(Graph& graph, std::vector<LabEdge>& edge_list,
                      const DFSCode& DFS_CODE, History& history) {
  std::map<VeridT, VeridT> vertex_id_map;
  for (size_t i = 0; i < history.size(); i++) {
    if (vertex_id_map.count(DFS_CODE[i].from) == 0)
      vertex_id_map[DFS_CODE[i].from] = history[i]->from;
    if (vertex_id_map.count(DFS_CODE[i].to) == 0)
      vertex_id_map[DFS_CODE[i].to] = history[i]->to;
  }
  // now add the backward edge using the last entry of the DFS code
  VeridT from = vertex_id_map[DFS_CODE.back().from];
  VeridT to   = vertex_id_map[DFS_CODE.back().to];
  for (auto it : graph.edges(from)) {
    GNode dst = graph.getEdgeDst(it);
    if (dst == to)
      return &(edge_list[*it]);
  }
  return 0;
}

/////////////////////////////////////////////////
// Canonical check (minimal DFSCode) functions //
/////////////////////////////////////////////////
bool get_forward_rmpath(CGraph& graph, LabEdge* e, LabelT minlabel,
                        History& history, EdgeList& result) {
  result.clear();
  assert(e->to >= 0 && e->to < graph.size());
  assert(e->from >= 0 && e->from < graph.size());
  LabelT tolabel = graph[e->to].label;
  for (Vertex::const_edge_iterator it = graph[e->from].edge.begin();
       it != graph[e->from].edge.end(); ++it) {
    LabelT tolabel2 = graph[it->to].label;
    if (e->to == it->to || minlabel > tolabel2 || history.hasVertex(it->to))
      continue;
    if (e->elabel < it->elabel ||
        (e->elabel == it->elabel && tolabel <= tolabel2))
      result.push_back(const_cast<LabEdge*>(&(*it)));
  }
  return (!result.empty());
}

bool get_forward_pure(CGraph& graph, LabEdge* e, LabelT minlabel,
                      History& history, EdgeList& result) {
  result.clear();
  assert(e->to >= 0 && e->to < graph.size());
  // Walk all edges leaving from vertex e->to.
  for (Vertex::const_edge_iterator it = graph[e->to].edge.begin();
       it != graph[e->to].edge.end(); ++it) {
    assert(it->to >= 0 && it->to < graph.size());
    if (minlabel > graph[it->to].label || history.hasVertex(it->to))
      continue;
    result.push_back(const_cast<LabEdge*>(&(*it)));
  }
  return (!result.empty());
}

bool get_forward_root(CGraph& graph, const Vertex& v, EdgeList& result) {
  result.clear();
  for (Vertex::const_edge_iterator it = v.edge.begin(); it != v.edge.end();
       ++it) {
    assert(it->to >= 0 && it->to < graph.size());
    if (v.label <= graph[it->to].label)
      result.push_back(const_cast<LabEdge*>(&(*it)));
  }
  return (!result.empty());
}

LabEdge* get_backward(CGraph& graph, LabEdge* e1, LabEdge* e2,
                      History& history) {
  if (e1 == e2)
    return 0;
  assert(e1->from >= 0 && e1->from < graph.size());
  assert(e1->to >= 0 && e1->to < graph.size());
  assert(e2->to >= 0 && e2->to < graph.size());
  for (Vertex::const_edge_iterator it = graph[e2->to].edge.begin();
       it != graph[e2->to].edge.end(); ++it) {
    if (history.hasEdge(it->id))
      continue;
    if ((it->to == e1->from) &&
        ((e1->elabel < it->elabel) ||
         ((e1->elabel == it->elabel) &&
          (graph[e1->to].label <= graph[e2->to].label)))) {
      return const_cast<LabEdge*>(&(*it));
    } // if(...)
  }   // for(it)
  return 0;
}

#endif
