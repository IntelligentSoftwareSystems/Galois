#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>

#include <mpi.h>

#include "galois/runtime/Network.h"
//#include "galois/runtime/Barrier.h"
#include "galois/runtime/Substrate.h"
#include "galois/Timer.h"
#include "galois/Timer.h"

using namespace galois::runtime;

// tests small message bandwidth

static std::atomic<int> num_recv;

void func(uint32_t, RecvBuffer&) {
  //  std::cerr << "!";
  ++num_recv;
}

int main(int argc, char** argv) {
  galois::StatManager s;

  using galois::start_now;

  int trials = 1000000;
  if (argc > 1)
    trials = atoi(argv[1]);

  NetworkInterface& net = getSystemNetworkInterface();
  // galois::StatManager sm;

  std::cerr << "I am " << net.ID << " using " << net.Num << "\n";
  auto tnum = trials * (net.Num - 1);
  std::cerr << "Sending " << trials << " between every host.  " << tnum
            << " total messages per host.\n";

  for (int num = 0; num < 20; num += 2) {
    num_recv = 0;
    galois::runtime::getHostBarrier().wait();

    std::vector<uint32_t> payload(num);

    galois::Timer T(start_now), T2(start_now), T3(start_now);
    for (int j = 0; j < net.Num; ++j) {
      if (j != net.ID) {
        for (int i = 0; i < trials; ++i) {
          SendBuffer buf;
          gSerialize(buf, payload);
          net.sendMsg(j, func, buf);
        }
      }
    }
    net.flush();
    T3.stop();
    while (num_recv < trials * (net.Num - 1)) {
      net.handleReceives();
    }
    T2.stop();
    galois::runtime::getHostBarrier().wait();
    T.stop();
    std::stringstream os;
    auto t  = T.get();
    auto t2 = T2.get();
    auto t3 = T3.get();
    os << net.ID << "@" << num << ":\tmessages " << t << " ms " << tnum << " "
       << (double)tnum / t << " msg/ms\tpayload "
       << num * sizeof(uint32_t) * tnum << " B "
       << (num * sizeof(uint32_t) * tnum) / t << " B/ms\n";
    os << "\tmessages " << t2 << " ms " << tnum << " " << (double)tnum / t2
       << " msg/ms\tpayload " << num * sizeof(uint32_t) * tnum << " B "
       << (num * sizeof(uint32_t) * tnum) / t2 << " B/ms\n";
    os << "\tSB " << net.reportSendBytes() << " SM " << net.reportSendMsgs()
       << " RB " << net.reportRecvBytes() << " RM " << net.reportRecvMsgs()
       << " SB/M " << (double)net.reportSendBytes() / net.reportSendMsgs()
       << " RB/M " << (double)net.reportRecvBytes() / net.reportRecvMsgs()
       << "\n";
    os << "\tTotal Time " << t << " Without Barrier " << t2 << " Send Only "
       << t3 << "\n";
    auto v = net.reportExtra();
    os << "\tTimeout " << v[0] << " Overflow " << v[1] << " Urgent " << v[2]
       << " Enqueued " << v[3] << " Dequeued " << v[4] << "\n";
    os << "\tSend B/pkg " << net.reportSendBytes() / (double)v[3]
       << " Send M/pkg " << net.reportSendMsgs() / (double)v[3]
       << " Recv B/pkg " << net.reportRecvBytes() / (double)v[4]
       << " Recv M/pkg " << net.reportRecvMsgs() / (double)v[4] << "\n";
    std::cout << os.str();
  }

  std::cerr << "I am " << net.ID << " using " << net.Num << "\n";
  std::cerr << "Sending " << trials << " between every host.  " << tnum
            << " total messages per host.\n";

  for (int num = 0; num < 20; num += 2) {
    num_recv = 0;
    galois::runtime::getHostBarrier().wait();
    tnum = num * 1024;
    std::vector<uint32_t> payload(tnum, 1);

    galois::Timer T(start_now), T2(start_now), T3(start_now);
    for (int j = 0; j < net.Num; ++j) {
      if (j != net.ID) {
        for (int i = 0; i < 10; ++i) {
          SendBuffer buf;
          gSerialize(buf, payload);
          net.sendTagged(j, 1, buf);
        }
      }
    }
    net.flush();
    T3.stop();
    for (int j = 0; j < net.Num; ++j) {
      if (j != net.ID) {
        decltype(net.recieveTagged(1, nullptr)) d;
        for (int i = 0; i < 10; ++i) {
          do {
            d = net.recieveTagged(1, nullptr);
          } while (!d);
        }
      }
    }
    T2.stop();
    galois::runtime::getHostBarrier().wait();
    T.stop();
    std::stringstream os;
    tnum *= (net.Num - 1);
    tnum *= sizeof(uint32_t) * 10;
    auto t  = T.get();
    auto t2 = T2.get();
    auto t3 = T3.get();
    if (t > 0) {
      os << net.ID << "@" << num << ":\tmessages " << t << " ms "
         << "\tpayload " << tnum << " B " << tnum / t << " B/ms\n";
      os << "\tSB " << net.reportSendBytes() << " SM " << net.reportSendMsgs()
         << " RB " << net.reportRecvBytes() << " RM " << net.reportRecvMsgs()
         << " SB/M " << (double)net.reportSendBytes() / net.reportSendMsgs()
         << " RB/M " << (double)net.reportRecvBytes() / net.reportRecvMsgs()
         << "\n";
      os << "\tTotal Time " << t << " Without Barrier " << t2 << " Send Only "
         << t3 << "\n";
      auto v = net.reportExtra();
      os << "\tTimeout " << v[0] << " Overflow " << v[1] << " Urgent " << v[2]
         << " Enqueued " << v[3] << " Dequeued " << v[4] << "\n";
      os << "\tSend B/pkg " << net.reportSendBytes() / (double)v[3]
         << " Send M/pkg " << net.reportSendMsgs() / (double)v[3]
         << " Recv B/pkg " << net.reportRecvBytes() / (double)v[4]
         << " Recv M/pkg " << net.reportRecvMsgs() / (double)v[4] << "\n";
    } else {
      os << net.ID << "@" << num << "\tNo Time?\n";
    }
    std::cout << os.str();
  }

  galois::runtime::reportStat(nullptr, "NetSB", net.reportSendBytes(), 0);
  galois::runtime::reportStat(nullptr, "NetSM", net.reportSendMsgs(), 0);
  galois::runtime::reportStat(nullptr, "NetRB", net.reportRecvBytes(), 0);
  galois::runtime::reportStat(nullptr, "NetRM", net.reportRecvMsgs(), 0);
  galois::runtime::getHostBarrier().wait();
  return 0;
}
