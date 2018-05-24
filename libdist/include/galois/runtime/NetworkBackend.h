#ifndef GALOIS_RUNTIME_NETWORKBACKEND_H
#define GALOIS_RUNTIME_NETWORKBACKEND_H

#include "galois/substrate/SimpleLock.h"
#include <cstdint>
#include <boost/intrusive/list.hpp>

namespace galois {
namespace runtime {

class NetworkBackend {
public:
  struct SendBlock : public boost::intrusive::list_base_hook<> {
    SendBlock(unsigned char* d) :dest(~0), size(0), data(d) {}
    uint32_t dest, size, tag;
    unsigned char* data;
  };

  typedef boost::intrusive::list<SendBlock> BlockList;

protected:
  uint32_t sz, _ID, _Num;
  substrate::SimpleLock flLock;
  BlockList freelist;

  NetworkBackend(unsigned size);

public:
  virtual ~NetworkBackend();

  SendBlock* allocSendBlock();
  void freeSendBlock(SendBlock*);

  //! send a block.  data is now owned by the Backend
  virtual void send(SendBlock* data) = 0;
  
  //! recieve a message, data is owned by the caller
  //1 and must be returned to this class
  virtual SendBlock* recv() = 0;

  //! make progress
  virtual void flush(bool block = false) = 0;

  //! returns size used by network
  uint32_t size() const { return sz; }

  uint32_t ID() const { return _ID; }
  uint32_t Num() const { return _Num; }

};


NetworkBackend& getSystemNetworkBackend();

}
}

#endif
