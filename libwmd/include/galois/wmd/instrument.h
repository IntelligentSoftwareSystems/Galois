#ifndef WMD_INSTRUMENT_H_
#define WMD_INSTRUMENT_H_

#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>

#include "galois/AtomicWrapper.h"
#include "galois/DynamicBitset.h"
#include "galois/DReducible.h"

namespace agile::workflow1 {

class Instrument;

inline std::unique_ptr<Instrument> instrument;

#ifdef GALOIS_INSTRUMENT
#define I_INIT(GRAPH_NAME, HOST, NUM_HOSTS, NUM_EDGES)                         \
  ({                                                                           \
    agile::workflow1::Instrument::init(GRAPH_NAME, HOST, NUM_HOSTS,            \
                                       NUM_EDGES);                             \
  })
#define I_DEINIT()                                                             \
  { agile::workflow1::instrument = nullptr; }
#define I_NEW_FILE(NAME_SUFFIX, NUM_EDGES)                                     \
  ({ agile::workflow1::instrument->new_file(NAME_SUFFIX, NUM_EDGES); })
#define I_ROUND(ROUND_NUM)                                                     \
  ({ agile::workflow1::instrument->log_round(ROUND_NUM); })
#define I_CLEAR() ({ agile::workflow1::instrument->clear(); })
#define I_RS() ({ agile::workflow1::instrument->record_local_read_stream(); })
#define I_RR(MIRROR)                                                           \
  ({ agile::workflow1::instrument->record_read_random(MIRROR); })
#define I_WR(MIRROR)                                                           \
  ({ agile::workflow1::instrument->record_write_random(MIRROR); })
#define I_WRR(REMOTE_HOST)                                                     \
  ({ agile::workflow1::instrument->record_write_random_remote(REMOTE_HOST); })
#define I_WM(WRITES) ({ agile::workflow1::instrument->write_many(WRITES); })
#define I_LC(REMOTE_HOST, BYTES)                                               \
  ({ agile::workflow1::instrument->log_communication(REMOTE_HOST, BYTES); })
#define I_BM(NODES)                                                            \
  ({ agile::workflow1::instrument->broadcast_masters(NODES); })
#else
#define I_INIT(GRAPH_NAME, HOST, NUM_HOSTS, NUM_EDGES) ({})
#define I_DEINIT() ({})
#define I_NEW_FILE(NAME_SUFFIX) ({})
#define I_ROUND(ROUND_NUM) ({})
#define I_CLEAR() ({})
#define I_RS() ({})
#define I_RR(MIRROR) ({})
#define I_WR(MIRROR) ({})
#define I_WRR(REMOTE_HOST) ({})
#define I_WM(WRITES) ({})
#define I_LC(REMOTE_HOST, BYTES) ({})
#define I_BM(NODES) ({})
#endif

class Instrument {
public:
  uint64_t hostID;
  uint64_t numHosts;
  std::string graph_name;

  std::unique_ptr<galois::DGAccumulator<uint64_t>> local_read_stream;
  std::unique_ptr<galois::DGAccumulator<uint64_t>> master_read;
  std::unique_ptr<galois::DGAccumulator<uint64_t>> master_write;
  std::unique_ptr<galois::DGAccumulator<uint64_t>> mirror_read;
  std::unique_ptr<galois::DGAccumulator<uint64_t>> mirror_write;
  std::unique_ptr<galois::DGAccumulator<uint64_t>[]> remote_comm_to_host;
  std::ofstream file;

  static void init(const std::string& graph_name_, uint64_t hid, uint64_t numH,
                   uint64_t numEdges) {
    instrument = std::make_unique<Instrument>(graph_name_, hid, numH, numEdges);
  }

  Instrument(const std::string& graph_name_, uint64_t hid, uint64_t numH,
             uint64_t numEdges) {
    hostID     = hid;
    numHosts   = numH;
    graph_name = graph_name_;

    local_read_stream = std::make_unique<galois::DGAccumulator<uint64_t>>();
    master_read       = std::make_unique<galois::DGAccumulator<uint64_t>>();
    master_write      = std::make_unique<galois::DGAccumulator<uint64_t>>();
    mirror_read       = std::make_unique<galois::DGAccumulator<uint64_t>>();
    mirror_write      = std::make_unique<galois::DGAccumulator<uint64_t>>();
    remote_comm_to_host =
        std::make_unique<galois::DGAccumulator<uint64_t>[]>(numH);
    clear();

    // start instrumentation
    file.open(graph_name + "_" + std::to_string(numH) + "procs_id" +
              std::to_string(hid));
    file << "#####   Stat   #####" << std::endl;
    file << "host " << hid << " total edges: " << numEdges << std::endl;
  }

  void new_file(const std::string& filename_extension, uint64_t numEdges) {
    file.close();
    file.open(graph_name + filename_extension + "_" + std::to_string(numHosts) +
              "procs_id" + std::to_string(hostID));
    file << "#####   Stat   #####" << std::endl;
    file << "host " << hostID << " total edges: " << numEdges << std::endl;
  }

  void clear() {
    local_read_stream->reset();
    master_read->reset();
    master_write->reset();
    mirror_read->reset();
    mirror_write->reset();

    for (auto i = 0ul; i < numHosts; i++) {
      remote_comm_to_host[i].reset();
    }
  }

  void record_local_read_stream() { *local_read_stream += 1; }

  void record_read_random(bool mirror = false) {
    if (!mirror) { // master
      *master_read += 1;
    } else { // mirror
      *mirror_read += 1;
    }
  }

  void record_write_random(bool mirror = false) {
    if (!mirror) { // master
      *master_write += 1;
    } else { // mirror
      *mirror_write += 1;
    }
  }

  void record_write_random_remote(uint64_t remote_host) {
    *mirror_write += 1;
    remote_comm_to_host[remote_host] += 1;
  }

  void write_many(uint64_t accesses) { *master_write += accesses; }

  void log_run(uint64_t run) {
    file << "#####   Run " << run << "   #####" << std::endl;
  }

  void log_round(uint64_t num_iterations) {
    auto host_id   = hostID;
    auto num_hosts = numHosts;
    file << "#####   Round " << num_iterations << "   #####" << std::endl;
    file << "host " << host_id
         << " local read (stream): " << local_read_stream->read_local()
         << std::endl;
    file << "host " << host_id << " master reads: " << master_read->read_local()
         << std::endl;
    file << "host " << host_id
         << " master writes: " << master_write->read_local() << std::endl;
    file << "host " << host_id << " mirror reads: " << mirror_read->read_local()
         << std::endl;
    file << "host " << host_id
         << " mirror writes: " << mirror_write->read_local() << std::endl;

    for (uint32_t i = 0; i < num_hosts; i++) {
      file << "host " << host_id << " remote communication for host " << i
           << ": " << remote_comm_to_host[i].read_local() << std::endl;
    }
  }

  void log_communication(uint64_t target_host, uint64_t bytes) {
    remote_comm_to_host[target_host] += bytes;
  }

  void broadcast_masters(uint64_t num_nodes) {
    auto host_id   = hostID;
    auto num_hosts = numHosts;

    for (uint64_t h = 0; h < num_hosts; h++) {
      if (h == host_id) {
        continue;
      }
      remote_comm_to_host[h] += num_nodes / num_hosts;
    }
  }
};

} // namespace agile::workflow1

#endif
