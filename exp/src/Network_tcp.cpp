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
#include <iostream>

using namespace Galois::Runtime;

static const bool debugRecv = true;
static __thread bool inRecv = false;

namespace {

class NetworkInterfaceTCP : public NetworkInterface {

  struct header {
    uint32_t size;
    uintptr_t func;
  };

  struct recvIP {
    int socket;
    LL::SimpleLock lock;
    std::deque<unsigned char> buffer;

    void doRead() {
      std::lock_guard<LL::SimpleLock> lg(lock);
      unsigned char data[256];
      int r = read(socket, data, sizeof(data));
      if (r > 0)
        std::copy(&data[0], &data[r], std::back_inserter(buffer));
    }

    bool recvMessage(recvFuncTy& recv, RecvBuffer& buf) {
      std::lock_guard<LL::SimpleLock> lg(lock);
      if (buffer.size() < sizeof(header))
        return false;
      header h;
      gDeserializeRaw(buffer.begin(), h);
      if (h.size - sizeof(header) >= buffer.size()) {
        recv = (recvFuncTy)h.func;
        buf.reset(h.size);
        std::copy(buffer.begin()+sizeof(header), buffer.begin() + h.size + sizeof(header),
                  (unsigned char*)buf.linearData());
        buffer.erase(buffer.begin(), buffer.begin() + h.size + sizeof(header));
        return true;
      }
      return false;
    }
  };

  struct sendIP {
    int socket;
    LL::SimpleLock lock;
    std::deque<unsigned char> buffer;

    void doSend() {
      std::lock_guard<LL::SimpleLock> lg(lock);
      if (!buffer.empty()) {
        unsigned char data[256];
        int num = std::copy(buffer.begin(), Galois::safe_advance(buffer.begin(), buffer.end(), sizeof(data)), data) - data;
        int r = write(socket, data, num);
        if (r > 0)
          buffer.erase(buffer.begin(), buffer.begin() + r);
      }
    }

    void sendMessage(recvFuncTy recv, SendBuffer& buf) {
      std::lock_guard<LL::SimpleLock> lg(lock);
      header h = {(uint32_t)buf.size(), (uintptr_t)recv};
      unsigned char* ch = (unsigned char*)&h;
      std::copy(ch, ch + sizeof(header), std::back_inserter(buffer));
      std::copy((unsigned char*)buf.linearData(), (unsigned char*)buf.linearData() + buf.size(), std::back_inserter(buffer));
    }
  };

  //socket per host
  //this host's listen socket is at it's offset
  //the remainder are client sockets
  std::vector<int> sockets;
  std::vector<sockaddr_in> clients;
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

  void connectToMaster(std::string& strportno) {
    sockets.resize(1);

    std::string master;
    if (!LL::EnvCheck("GALOIS_TCP_HOST", master))
      abort();
    
    struct addrinfo *result;
    struct addrinfo hint;
    bzero(&hint, sizeof(hint));
    hint.ai_family = AF_UNSPEC;
    hint.ai_socktype = SOCK_STREAM;
    hint.ai_flags = AI_NUMERICSERV;
    if (getaddrinfo(master.c_str(), strportno.c_str(), &hint, &result) < 0)
      error("ERROR in getaddrinfo");

    sockets[0] = -1;
    for (addrinfo* rp = result; rp != NULL; rp = rp->ai_next) {
      if ((sockets[0] = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol)) == -1)
        continue;
      if (connect(sockets[0],rp->ai_addr, rp->ai_addrlen) != -1)
        break; // success
      close(sockets[0]);
      sockets[0] = -1;
    }

    freeaddrinfo(result);

    if (sockets[0] == -1)
      error("ERROR connecting");

    setNoDelay(sockets[0]);
  }

  int setupListenSocket(int portno, std::string& strportno) {
    int listensocket = socket(AF_INET, SOCK_STREAM, 0);
    if (listensocket < 0)
      error("ERROR opening socket");
    setNoDelay(listensocket);
    sockaddr_in serv_addr;
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(listensocket, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
      error("ERROR on binding");
    if (listen(listensocket, numhosts) < 0)
      error("ERROR on listening");
    std::cerr << "Listening on " << portno << "(as " << strportno << ")\n";
    return listensocket;
  }

  void connectToChildren(int listensocket) {
    id = 0;
    sockets.resize(numhosts);
    clients.resize(numhosts);

    //connect all clients
    int waiting = numhosts - 1;
    while (waiting) {
      socklen_t clilen = sizeof(sockaddr_in);
      sockets[waiting] = accept(listensocket, (struct sockaddr *) &clients[waiting], &clilen);
      if (sockets[waiting] < 0)
        error("ERROR accepting");
      //tell client their id
      if (write(sockets[waiting], &waiting, sizeof(int)) < 0)
        error("ERROR in writing id");
      if (write(sockets[waiting], &numhosts, sizeof(int)) < 0)
        error("ERROR in writing humhosts");
      std::cerr << "heard from " << waiting << " of " << numhosts << "\n";
      --waiting;
    }
  }

  void connectToPeers(int listensocket, int portno) {
    sockets.resize(numhosts);
    for (int i = 1; i < numhosts; ++i) {
      if (i == id) { //connect
        for (int x = 1; x < numhosts; ++x) {
          if (x != id) {
            sockets[x] = socket(AF_INET, SOCK_STREAM, 0);
            if (sockets[x] < 0)
              error("ERROR opening socket");
            clients[x].sin_port = htons(portno);
            if (connect(sockets[x],(struct sockaddr *) &clients[x], sizeof(sockaddr_in)) < 0)
              error("ERROR connecting");
          }
        }
      } else { // listen
        socklen_t clilen = sizeof(sockaddr_in);
        sockets[i] = accept(listensocket, (struct sockaddr *) &clients[i], &clilen);
      }
    }
  }
        
public:
  NetworkInterfaceTCP() {
    int listensocket;
    int portno;
    std::string strportno;

    if (!LL::EnvCheck("GALOIS_TCP_PORT", portno))
      portno = 95863;
    if (!LL::EnvCheck("GALOIS_TCP_PORT", strportno))
      strportno = "95863";

    listensocket = setupListenSocket(portno, strportno);

    if (LL::EnvCheck("GALOIS_TCP_MASTER", numhosts)) {
      //get connection to all children
      connectToChildren(listensocket);
      //send address to everyone
      for (int i = 1; i < numhosts; ++i)
        if (write(sockets.at(i), &clients.at(0) , sizeof(sockaddr_in) * numhosts) < 0)
          error("ERROR in writing clientlist");
      //carry on
    } else { //client
      connectToMaster(strportno);
      if (read(sockets[0], &id, sizeof(int)) < sizeof(int))
        error("ERROR reading id");
      if (read(sockets[0], &numhosts, sizeof(int)) < sizeof(int))
        error("ERROR reading numhosts");
      std::cerr << "Read " << id << " of " << numhosts << "\n";

      //read address of everyone
      clients.resize(numhosts);
      if (read(sockets.at(0), &clients.at(0), sizeof(sockaddr_in) * numhosts) < 0)
        error("ERROR in reading clientlist");
      //connect to everyone
      connectToPeers(listensocket, portno);
    }

    recvState.resize(numhosts);
    sendState.resize(numhosts);
    FD_ZERO(&readfds);
    for (int x = 0; x < numhosts; ++x) {
      recvState[x].socket = sendState[x].socket = sockets[x];
      if (x != id)
        FD_SET(sockets.at(x), &readfds);
    }
    maxsocket = *std::max_element(sockets.begin(), sockets.end());

    if (close(listensocket) < 0)
      error("ERROR closing setup socket");

    ID = id;
    Num = numhosts;

    std::cerr << "I am " << ID << " of " << Num << "\n";
  }

  virtual ~NetworkInterfaceTCP() {
    for (int i = 0; i < numhosts; ++i)
      close(sockets.at(i));
  }

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    sendState[dest].sendMessage(recv, buf);
  }

  virtual bool handleReceives() {
    if (debugRecv)
      assert(!inRecv);
    fd_set rcheck = readfds;
    fd_set wcheck = readfds;
    timeval t = {0,0};
    int retval = select(1 + maxsocket, &rcheck, &wcheck, NULL, &t);
    if (retval == -1)
      error("ERROR in select");
    if (retval) {
      for (int x = 0; x < numhosts; ++x) {
        if (x != id) {
          if (FD_ISSET(sockets.at(x), &wcheck))
            sendState[x].doSend();
          if (FD_ISSET(sockets.at(x), &rcheck)) {
            recvIP& state = recvState[x];
            state.doRead();
            //keep lock for recv, since it only blocks other threads from recieves from that host
            recvFuncTy func;
            RecvBuffer buf(0);
            if (state.recvMessage(func, buf)) {
              if (debugRecv)
                inRecv = true;
              func(buf);
              if (debugRecv)
                inRecv = false;
            }
          }
        }
      }
    }
    return retval;
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
