/** Galois Network Layer for TCP -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/ll/EnvCheck.h"
#include "Galois/gstl.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <fcntl.h>

#include <iostream>

using namespace Galois::Runtime;

static const bool debugRecv = true;
static __thread bool inRecv = false;

namespace {

class NetworkInterfaceTCP : public NetworkInterface {

  struct recvIP {
    //for array expansion
    recvIP(const recvIP& other) :socket(other.socket) {}
    recvIP() :socket(-1) {}

    int socket;
    LL::SimpleLock lock;
    std::deque<unsigned char> buffer;
    std::atomic<bool> waiting;

    void doRead() {
      if (lock.try_lock()) {
        std::lock_guard<LL::SimpleLock> lg(lock, std::adopt_lock);
        unsigned char data[256];
        int r = 0;
        do {
          r = read(socket, data, sizeof(data));
          //std::cout << "R(" << r << ")";
          if (r > 0)
            std::copy(&data[0], &data[r], std::back_inserter(buffer));
        } while (r == sizeof(data));
        if (buffer.size())
          waiting = true;
      }
    }

    //Assumes lock is held by caller (needed for message in-order delivery)
    RecvBuffer recvMessage() {
      //std::lock_guard<LL::SimpleLock> lg(lock);
      if (buffer.size() < sizeof(uint32_t))
        goto exit_recv;

      uint32_t h;
      gDeserializeRaw(buffer.begin(), h);
      if (h + sizeof(uint32_t) <= buffer.size()) {
        RecvBuffer b(buffer.begin() + sizeof(uint32_t), 
                     buffer.begin() + sizeof(uint32_t) + h);
        buffer.erase(buffer.begin(), buffer.begin() + sizeof(uint32_t) + h);
        return b;
      }
    exit_recv:
      waiting = false;
      return RecvBuffer{};
    }

    bool pending() {
      return waiting;
    }
      
  };

  struct sendIP {
    //for array expansion
    sendIP(const sendIP& other) :socket(other.socket) {}
    sendIP() :socket(-1) {}

    int socket;
    LL::SimpleLock lock;
    std::vector<unsigned char> buffer;
    std::atomic<bool> waiting;

    void doWriteImpl() {
      //lock already held
      if (buffer.size()) {
        int r = write(socket, &buffer[0], buffer.size());
        if (r > 0)
          buffer.erase(buffer.begin(), buffer.begin() + r);
      }
      if (buffer.size() == 0)
        waiting = false;
    }
    
    void doWrite() {
      if (lock.try_lock()) {
        std::lock_guard<LL::SimpleLock> lg(lock, std::adopt_lock);
        doWriteImpl();
      }
    }
    
    void sendMessage(recvFuncTy recv, SendBuffer& buf) {
      std::lock_guard<LL::SimpleLock> lg(lock);
      auto insertPoint = std::back_inserter(buffer);
      //frame
      uint32_t h = buf.size() + sizeof(recvFuncTy);
      unsigned char* ch = (unsigned char*)&h;
      std::copy(ch, ch + sizeof(uint32_t), insertPoint);
      //prefix
      ch = (unsigned char*)&recv;
      std::copy(ch, ch + sizeof(recvFuncTy), insertPoint);
      //message
      std::copy((unsigned char*)buf.linearData(), (unsigned char*)buf.linearData() + buf.size(), insertPoint);
      if (!waiting)
        waiting = true;
      doWriteImpl();
    }

    bool pending() {
      return waiting;
    }
  };

  //socket per host
  //this host's listen socket is at it's offset
  //the remainder are client sockets
  std::vector<int> sockets;
  std::vector<recvIP> recvState;
  std::vector<sendIP> sendState;

  int id;
  int numhosts;
  fd_set readfds;
  int maxsocket;

  void error(const char* mesg) {
    perror(mesg);
    abort();
  }

  void setNoDelay(int socket) {
    int flag = 1;
    int ret = setsockopt( socket, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(flag) );
    if (ret == -1)
      std::cout << "Couldn't setsockopt(TCP_NODELAY)\n";
  }

  void setNonBlock(int socket) {
    int flags;
    flags = fcntl(socket,F_GETFL,0);
    assert(flags != -1);
    fcntl(socket, F_SETFL, flags | O_NONBLOCK);
  }

  int connectTo(std::string name, int port) {
    std::ostringstream os;
    os << port;
    return connectTo(name, os.str());
  }

  int connectTo(std::string name, std::string port) {
    struct addrinfo *result;
    struct addrinfo hint;
    bzero(&hint, sizeof(hint));
    hint.ai_family = AF_UNSPEC;
    hint.ai_socktype = SOCK_STREAM;
    hint.ai_flags = AI_NUMERICSERV;
    if (getaddrinfo(name.c_str(), port.c_str(), &hint, &result) < 0)
      error("ERROR in getaddrinfo");
    
    int sock = -1;
    for (addrinfo* rp = result; rp != NULL; rp = rp->ai_next) {
      if ((sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol)) == -1)
        continue;
      if (connect(sock,rp->ai_addr, rp->ai_addrlen) != -1)
        break; // success
      close(sock);
      sock = -1;
    }

    freeaddrinfo(result);
    
    if (sock == -1)
      error("ERROR connecting");

    setNoDelay(sock);
    return sock;
   }

  std::pair<int, std::vector<std::pair<std::string, int> > >
  getPeers(int listenport) {
    std::vector<std::pair<std::string, int> > peers;

    std::string serverport;
    if (!LL::EnvCheck("GALOIS_TCP_PORT", serverport))
      error("Error, no server port");
    std::string servername;
    if (!LL::EnvCheck("GALOIS_TCP_NAME", servername))
      error("Error, no server name");

    int serversocket = connectTo(servername, serverport);

    char buffer[2048];
    bzero(buffer, sizeof(buffer));
    
    int r = snprintf(buffer, sizeof(buffer), "%d\n", listenport);
    //std::cerr << buffer << "|" << r << "\n";
    if (write(serversocket, buffer, r) < r)
      error("ERROR in writing to starter");
    
    bzero(buffer, sizeof(buffer));
    int off = 0;
    while (!memchr(buffer, '\n', off)) {
      r = read(serversocket, &buffer[off], sizeof(buffer) - off);
      if (r < 0)
        error("ERROR reading from server");
      off += r;
    }
    buffer[--off] = '\0';

    std::vector<char*> things;
    things.push_back(buffer);
    char* tmp = buffer;
    while((tmp = (char*)memchr(tmp, ',', strlen(tmp)))) {
      *tmp = '\0';
      ++tmp;
      things.push_back(tmp);
    }
    // for(auto p : things)
    //   std::cerr << "|" << p << "|" << "\n";

    int num = atoi(things[0]);
    int me = atoi(things[1]);
    for (int i = 0; i < num; ++i) {
      std::string s(things.at(2+i*2));
      int port = atoi(things.at(3+i*2));
      peers.push_back(std::make_pair(s,port));
    }

    close(serversocket);

    return std::make_pair(me, peers);
  }

  std::pair<int,int> setupListenSocket() {
    int listensocket = socket(AF_INET, SOCK_STREAM, 0);
    if (listensocket < 0)
      error("ERROR opening socket");
    setNoDelay(listensocket);
    sockaddr_in serv_addr;
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = 0;
    if (bind(listensocket, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
      error("ERROR on binding");
    if (listen(listensocket, 128) < 0) //MAX unaccepted things
      error("ERROR on listening");

    struct sockaddr_in sin;
    socklen_t len = sizeof(sin);
    if (getsockname(listensocket, (struct sockaddr *)&sin, &len) == -1)
      perror("getsockname");
    else
      std::cerr << "Listening on " << ntohs(sin.sin_port) << "\n";
    return std::make_pair(listensocket, ntohs(sin.sin_port));
  }

  void connectToPeers(int listensocket, std::vector<std::pair<std::string, int> > hosts) {
    numhosts = hosts.size();
    sockets.resize(numhosts);
    for (int i = 0; i < id; ++ i)
      sockets[i] = accept4(listensocket, nullptr, 0, SOCK_NONBLOCK);
    //connect
    for (int x = id + 1; x < numhosts; ++x) {
      sockets[x] = connectTo(hosts[x].first, hosts[x].second);
      setNonBlock(sockets[x]);
    }
  }

  void tryIO() {
    //only check sockets with data
    fd_set wcheck;
    FD_ZERO(&wcheck);
    for (int x = 0; x < numhosts; ++x)
      if (sendState[x].pending())
        FD_SET(sockets.at(x), &wcheck);
    //check all readers
    fd_set rcheck = readfds;
    timeval t = {0,0}; //non-blocking
    int retval = select(1 + maxsocket, &rcheck, &wcheck, NULL, &t);
    if (retval == -1)
      error("ERROR in select");
    if (retval)
      for (int x = 0; x < numhosts; ++x) {
        if (FD_ISSET(sockets.at(x), &wcheck))
          sendState[x].doWrite();
        if (FD_ISSET(sockets.at(x), &rcheck))
          recvState[x].doRead();
      }
  }

public:
  NetworkInterfaceTCP() {
    int listensocket, listenport;
    std::tie(listensocket,listenport) = setupListenSocket();
    auto mepeers = getPeers(listenport);
    numhosts = mepeers.second.size();
    id = mepeers.first;
    connectToPeers(listensocket, mepeers.second);
    recvState.resize(numhosts);
    sendState.resize(numhosts);
    FD_ZERO(&readfds);
    assert(numhosts <= FD_SETSIZE);
    for (int x = 0; x < numhosts; ++x) {
      recvState[x].socket = sendState[x].socket = sockets[x];
      if (x != id) {
        assert(sockets.at(x) < FD_SETSIZE);
        FD_SET(sockets.at(x), &readfds);
      }
    }
    maxsocket = *std::max_element(sockets.begin(), sockets.end());

    if (close(listensocket) < 0)
      error("ERROR closing setup socket");

    ID = mepeers.first;
    Num = numhosts;

    std::cerr << "I am " << ID << " of " << Num << "\n";
  }

  virtual ~NetworkInterfaceTCP() {
    for (int i = 0; i < numhosts; ++i)
      if (i != ID)
        close(sockets.at(i));
  }

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    this->statSendNum += 1;
    this->statSendBytes += buf.size();
    sendState[dest].sendMessage(recv, buf);
  }

  virtual bool handleReceives() {
    if (debugRecv)
      assert(!inRecv);
    bool retval = false;
    tryIO();
    for (int x = 0; x < numhosts; ++x) {
      if (x != id && recvState[x].pending()) {
        std::lock_guard<LL::SimpleLock> lg(recvState[x].lock);
        RecvBuffer buf = recvState[x].recvMessage();
        if (buf.size()) {
          this->statRecvNum += 1;
          this->statRecvBytes += buf.size();
          retval = true;
          recvFuncTy func;
          gDeserialize(buf, func);
          if (debugRecv)
            inRecv = true;
          func(buf);
          if (debugRecv)
            inRecv = false;
        }
      }
    }
    return retval;
  }

  virtual void flush() {
    tryIO();
  }

  virtual bool needsDedicatedThread() {
    return false;
  }
};

}

#ifdef USE_TCP
NetworkInterface& Galois::Runtime::getSystemNetworkInterface() {
  static NetworkInterfaceTCP net;
  return net;
}
#endif
