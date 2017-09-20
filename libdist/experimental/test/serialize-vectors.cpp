#include "galois/Galois.h"
#include "galois/Runtime/Serialize.h"

#include <iostream>
#include <vector>
#include <algorithm>

using namespace galois::Runtime;

int main() {

  auto& net = galois::runtime::getSystemNetworkInterface();

  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> assigned_edges_perhost;
  std::vector<std::vector<uint64_t>> assigned_edges_perhost_linear;

  assigned_edges_perhost.resize(net.Num);
  uint64_t n = {0};
  for(auto h = 0; h < net.Num; ++h){
    assigned_edges_perhost[h].resize(1024*100);
    std::generate(assigned_edges_perhost[h].begin(), assigned_edges_perhost[h].end(), [&n]{return std::make_pair(n++, n);});

    std::cout << "["<<net.ID<<"] Size : " << assigned_edges_perhost[h].size() << "\n";
  }


  assigned_edges_perhost_linear.resize(net.Num);
  for(auto h = 0; h < net.Num; ++h){
    assigned_edges_perhost_linear[h].resize(2*1024*1024);
    std::generate(assigned_edges_perhost_linear[h].begin(), assigned_edges_perhost_linear[h].end(), [&n]{return n++;});

    std::cout << "["<<net.ID<<"] Size : " << assigned_edges_perhost_linear[h].size() << "\n";
  }


  for (unsigned x = 0; x < net.Num; ++x) {
    if(x == net.ID) continue;

    galois::runtime::SendBuffer b;
    std::cerr << "[" << net.ID<<"]" << " serialize start : " << x << "\n";
    gSerialize(b, assigned_edges_perhost_linear[x]);
    std::cerr << "[" << net.ID<<"]" << " serialize done : " << x << "\n";
    net.sendTagged(x, galois::runtime::evilPhase, b);
    assigned_edges_perhost_linear[x].clear();
    std::stringstream ss;
    ss <<" sending from : " <<  net.ID << " to : " << x << " Size should b4 ZERO : " << assigned_edges_perhost_linear[x].size()<< "\n";
    std::cout << ss.str() << "\n";
  }

  //receive
  for (unsigned x = 0; x < net.Num; ++x) {
    if(x == net.ID) continue;

    decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
    do {
      net.handleReceives();
      p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
    } while(!p);

    galois::runtime::gDeserialize(p->second, assigned_edges_perhost_linear[p->first]);
    std::stringstream ss;
    ss <<" received on : " <<  net.ID << " from : " << x << " Size : " << assigned_edges_perhost_linear[p->first].size()<< "\n";
    std::cout << ss.str();
  }
  ++galois::runtime::evilPhase;

}

