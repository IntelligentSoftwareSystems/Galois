#include "galois/Logging.h"
#include "galois/graphs/GNNGraph.h"

// actually does remapping
int main() {
  galois::DistMemSys G;
  galois::graphs::LC_CSR_Graph<char, void> orig;
  orig.readGraphFromGRFile(
      "/net/ohm/export/iss/inputs/Learning/ogbn-papers100M.tgr");
  // orig.readGraphFromGRFile("/net/ohm/export/iss/inputs/Learning/ogbn-papers100M.gr");

  std::vector<uint64_t> node_indices;
  node_indices.resize(orig.size(), 0);
  std::vector<uint32_t> destinations;
  destinations.resize(orig.sizeEdges(), 0);

  // get mapping
  std::string remap_name =
      galois::default_gnn_dataset_path + "ogbn-papers100M-remap-mapping.bin";
  std::ifstream file_stream;
  file_stream.open(remap_name, std::ios::binary | std::ios::in);
  std::vector<uint32_t> new_to_old(111059956);
  file_stream.read((char*)new_to_old.data(),
                   sizeof(uint32_t) * new_to_old.size());
  file_stream.close();

  std::vector<uint32_t> old_to_new(111059956);

  galois::DynamicBitSet mark_all;
  mark_all.resize(orig.size());
  mark_all.reset();

  // get # edges on each node in remapped
  galois::do_all(
      galois::iterate(orig.begin(), orig.end()), [&](uint32_t remapped_id) {
        uint32_t source_id    = new_to_old[remapped_id];
        old_to_new[source_id] = remapped_id;
        mark_all.set(source_id);
        GALOIS_LOG_ASSERT(source_id < orig.size());
        // TODO check duplicates too
        node_indices[remapped_id] =
            std::distance(orig.edge_begin(source_id), orig.edge_end(source_id));
      });

  galois::do_all(galois::iterate(0, 111059956),
                 [&](unsigned i) { GALOIS_LOG_ASSERT(mark_all.test(i)); });

  // prefix sum it
  for (size_t i = 1; i < node_indices.size(); i++) {
    node_indices[i] += node_indices[i - 1];
  }
  // write all edges
  galois::do_all(
      galois::iterate(orig.begin(), orig.end()),
      [&](uint32_t remapped_id) {
        uint32_t source_id = new_to_old[remapped_id];
        GALOIS_LOG_ASSERT(source_id < orig.size());
        uint64_t current_idx;
        if (remapped_id != 0) {
          current_idx = node_indices[remapped_id - 1];
        } else {
          current_idx = 0;
        }
        uint64_t my_end = node_indices[remapped_id];

        for (auto ei = orig.edge_begin(source_id);
             ei != orig.edge_end(source_id); ei++) {
          uint32_t dest               = old_to_new[orig.getEdgeDst(ei)];
          destinations[current_idx++] = dest;
        }
        GALOIS_LOG_ASSERT(current_idx == my_end);
        // TODO check duplicates too
        // node_indices[remapped_id] = std::distance(orig.edge_begin(node_id),
        // orig.edge_end(node_id));
      },
      galois::steal());

  // write everything
  struct Header {
    uint64_t version;
    uint64_t size;
    uint64_t numNodes;
    uint64_t numEdges;
  };
  Header h;
  h.version  = 1;
  h.size     = 0;
  h.numNodes = orig.size();
  h.numEdges = orig.sizeEdges();

  std::string filename =
      "/net/ohm/export/iss/inputs/Learning/ogbn-papers100M-remap.tgr";
  // std::string filename =
  // "/net/ohm/export/iss/inputs/Learning/ogbn-papers100M-remap.gr";
  std::ofstream write_stream;
  write_stream.open(filename, std::ios::binary | std::ios::out);
  write_stream.write((char*)&h, sizeof(Header));
  write_stream.write((char*)node_indices.data(),
                     sizeof(uint64_t) * node_indices.size());
  write_stream.write((char*)destinations.data(),
                     sizeof(uint32_t) * destinations.size());

  write_stream.close();
}
