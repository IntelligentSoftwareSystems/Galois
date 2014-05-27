/** Galois Distributed Loop -*- C++ -*-
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
 * @section Description
 *
 * Implementation of the Galois distributed foreach iterator.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_PARALLELWORKDIST_H
#define GALOIS_RUNTIME_PARALLELWORKDIST_H

#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/DistSupport.h"
#include "Galois/Runtime/PerHostStorage.h"

#include <iostream>
#include <iterator>

namespace Galois {
namespace Runtime {

template<typename WLTy, typename ItemTy, typename FunctionTy>
void for_each_landing_pad(RecvBuffer& buf) {
  //extract stuff
  FunctionTy f;
  std::deque<ItemTy> data;
  std::string loopname;
  gDeserialize(buf,f,data,loopname);

  //Start locally
  Galois::Runtime::for_each_impl<WLTy>(Galois::Runtime::makeStandardRange(data.begin(), data.end()), f, loopname.c_str());

  // place a MPI barrier here for all the hosts to synchronize
  //net.systemBarrier();
}

//:Gill: new landing pad.
template<typename WLTy, typename ItemTy, typename FunctionTy, typename ValueTy>
void for_each_landing_pad(RecvBuffer& buf) {
  //extract stuff
  FunctionTy f;
  std::deque<ItemTy> data;
  std::string loopname;
  ValueTy iterations, node_num, total_nodes;
  gDeserialize(buf,iterations, node_num, total_nodes,f,data,loopname);

  NetworkInterface& net = getSystemNetworkInterface();

   uint32_t blockSize = node_num/networkHostNum; //user nodes divided for sgd.
   std::cout << "starting for loop on host = " << networkHostID  << "\n";
//	net.systemBarrier();
   for(int i = 0; i < iterations; ++i) {
	uint32_t blockNum = (networkHostID + i)%networkHostNum;
	uint32_t startRange = blockSize*blockNum;
	uint32_t endRange = blockSize*(blockNum + 1);
	if(blockNum == (networkHostNum - 1)) endRange = (node_num);//+ 1); 
	
	startRange += (total_nodes + 1);
	endRange += (total_nodes );
	f.startRange = startRange;
	f.endRange = endRange;
	
	std::cout << "i" << i <<"[" << f.startRange << "," << f.endRange<< "]"  << " on host = " << networkHostID <<"\n";
	//Start locally
	Galois::Runtime::for_each_impl<WLTy>(Galois::Runtime::makeStandardRange(data.begin(), data.end()), f, loopname.c_str());
	net.systemBarrier();
    }

  //Start locally
  //Galois::Runtime::for_each_impl<WLTy>(Galois::Runtime::makeStandardRange(data.begin(), data.end()), f, loopname.c_str());

  // place a MPI barrier here for all the hosts to synchronize
  net.systemBarrier();
}



template<typename WLTy, typename T, typename FunctionTy>
void for_each_local_landing_pad(RecvBuffer& buf) {
  //extract stuff
  FunctionTy f;
  T data;
  std::string loopname;
  gDeserialize(buf,f,data,loopname);

  //Start locally
  Galois::Runtime::for_each_impl<WLTy>(Galois::Runtime::makeLocalRange(data), f, loopname.c_str());
  
  // place a MPI barrier here for all the hosts to synchronize
  //net.systemBarrier();
}

template<typename ItemTy, typename FunctionTy>
void do_all_landing_pad(RecvBuffer& buf) {
  //extract stuff
  FunctionTy f;
  std::deque<ItemTy> data;
  std::string lname;
  bool       steal;
  gDeserialize(buf,f,data,lname,steal);

  //Start locally
  do_all_impl(makeStandardRange(data.begin(), data.end()),f,lname.c_str(),steal);

  // place a MPI barrier here for all the hosts to synchronize
  //net.systemBarrier();
}

template<typename T, typename FunctionTy, typename ReducerTy>
void do_all_local_landing_pad(RecvBuffer& buf) {
  //extract stuff
  FunctionTy f;
  LocalRange<T> lr;
  std::string lname;
  bool steal;
  gDeserialize(buf,f,lr,lname,steal);

  //Start locally
  do_all_impl(lr, f, lname, steal);

  // place a MPI barrier here for all the hosts to synchronize
  //net.systemBarrier();
}

template<typename FunctionTy>
void on_each_impl_landing_pad(RecvBuffer& buf) {
  //extract stuff
  FunctionTy f;
  gDeserialize(buf,f);

  //Start locally
  on_each_impl(f);

  // place a MPI barrier here for all the hosts to synchronize
  //net.systemBarrier();
}

namespace {

template<typename WLTy, typename IterTy, typename FunctionTy>
void for_each_dist(StandardRange<IterTy> r, FunctionTy f, const char* loopname) {
  // Get a handle to the network interface
  //  Don't move as NetworkInterface::Num and NetworkInterface::ID have to be initialized first
  NetworkInterface& net = getSystemNetworkInterface();

  std::cout << "inside for_each_dist OLD  " << networkHostID  << "\n";
  typedef typename std::iterator_traits<IterTy>::value_type ItemTy;

  //fast path for non-distributed
  if (NetworkInterface::Num == 1) {
    for_each_impl<WLTy>(r,f,loopname);
    return;
  }

  //copy out all data
  std::deque<ItemTy> allData;
  allData.insert(allData.end(), r.begin(), r.end());

  std::string lname(loopname);

  for (unsigned i = 1; i < NetworkInterface::Num; i++) {
    auto blk = block_range(allData.begin(), allData.end(), i, NetworkInterface::Num);
    std::deque<ItemTy> data(blk.first, blk.second);
    SendBuffer buf;
    // serialize function and data
    gSerialize(buf,f,data, lname);
    //send data
    net.sendLoop (i, &for_each_landing_pad<WLTy,ItemTy,FunctionTy>, buf);
  }
  net.flush();
  net.handleReceives();
  //now get our data
  auto myblk = block_range(allData.begin(), allData.end(), 0, NetworkInterface::Num);

  //Start locally
  for_each_impl<WLTy>(Galois::Runtime::makeStandardRange(myblk.first, myblk.second), f, loopname);

  // place a MPI barrier here for all the hosts to synchronize
  // net.systemBarrier();
}


//Gill: New Variant to handle iteration number.
template<typename WLTy, typename IterTy, typename ValueTy, typename FunctionTy>
void for_each_dist(IterTy b, IterTy e, ValueTy iterations, ValueTy node_num, FunctionTy f, const char* loopname) {
  // Get a handle to the network interface
  //  Don't move as networkHostNum and networkHostID have to be initialized first
  NetworkInterface& net = getSystemNetworkInterface();

  std::cout << "inside for_each_dist  " << networkHostID  << "\n";
  typedef typename std::iterator_traits<IterTy>::value_type ItemTy;

  uint32_t total_nodes = std::distance(b,e); // total movie nodes to be divided.
  //std::cout << "new for_each_dist i = " << iterations << "\n"; 
  //fast path for non-distributed
  if (networkHostNum == 1) {
    std::deque<ItemTy> allData;
    allData.insert(allData.end(), b,e);
    std::cout << "should not go here\n";
    f.startRange +=(total_nodes + 1);
    f.endRange += (total_nodes + node_num);// + 1 );  

	std::cout << "i" << 0 <<"[" << f.startRange << "," << f.endRange<< "]"  << " on host = " << networkHostID <<"\n";
    for_each_impl<WLTy>(Galois::Runtime::makeStandardRange(allData.begin(), allData.end()),f,loopname);
    return;
  }

  //copy out all data
  std::deque<ItemTy> allData;
  allData.insert(allData.end(), b,e);

  std::string lname(loopname);

  for (unsigned i = 1; i < networkHostNum; i++) {
    auto blk = block_range(allData.begin(), allData.end(), i, networkHostNum);
    std::deque<ItemTy> data(blk.first, blk.second);
    SendBuffer buf;
    // serialize function and data
    gSerialize(buf,iterations,node_num,total_nodes,f,data, lname);
    //send data
    net.send (i, &for_each_landing_pad<WLTy,ItemTy,FunctionTy,ValueTy>, buf);
  }
  net.handleReceives();
  //now get our data
  auto myblk = block_range(allData.begin(), allData.end(), 0, networkHostNum);

    
   // Using Iteration number to call for_each_impl number of times.
   uint32_t blockSize = node_num/networkHostNum; //user nodes divided for sgd.
   std::cout << "starting for loop on host = " << networkHostID  << "\n";
//	net.systemBarrier();
   for(int i = 0; i < iterations; ++i) {
	uint32_t blockNum = (networkHostID + i)%networkHostNum;
	uint32_t startRange = blockSize*blockNum;
	uint32_t endRange = blockSize*(blockNum + 1);
	if(blockNum == (networkHostNum - 1)) endRange = (node_num);// + 1); 
	
	startRange += (total_nodes + 1);
	endRange += (total_nodes );
	f.startRange = startRange;
	f.endRange = endRange;
	
	std::cout << "i" << i <<"[" << f.startRange << "," << f.endRange<< "]"  << " on host = " << networkHostID <<"\n";
	//Start locally
	for_each_impl<WLTy>(Galois::Runtime::makeStandardRange(myblk.first, myblk.second),  f, loopname);
	net.systemBarrier();
    }
    std::cout<<"------------------------------------------\n";
  // place a MPI barrier here for all the hosts to synchronize
  net.systemBarrier();
}

template<typename WLTy, typename T, typename FunctionTy>
void for_each_dist(LocalRange<T>& r, FunctionTy f, const char* loopname) {
  // Get a handle to the network interface
  //  Don't move as NetworkInterface::Num and NetworkInterface::ID have to be initialized first
  NetworkInterface& net = getSystemNetworkInterface();

  //fast path for non-distributed
  if (NetworkInterface::Num == 1) {
    for_each_impl<WLTy>(r,f,loopname);
    return;
  }

  std::string lname(loopname);

  for (unsigned i = 1; i < NetworkInterface::Num; i++) {
    SendBuffer buf;
    // serialize function and data
    gSerialize(buf,f,r,lname);
    //send data
    net.sendLoop (i, &for_each_local_landing_pad<WLTy,T,FunctionTy>, buf);
  }
  net.flush();
  net.handleReceives();
  //Start locally
  for_each_impl<WLTy>(r, f, loopname);

  // place a MPI barrier here for all the hosts to synchronize
  //  net.systemBarrier();
}

template<typename T, typename FunctionTy, typename ReducerTy>
void do_all_impl_dist(LocalRange<T>& lr, FunctionTy f, ReducerTy r, bool needsReduce) {
  // Get a handle to the network interface
  NetworkInterface& net = getSystemNetworkInterface();

  // Should be called only outside a for_each for now
  assert(!inGaloisForEach);

  //fast path for non-distributed
  if (NetworkInterface::Num == 1) {
    do_all_impl(lr,f,r,needsReduce);
    return;
  }

  for (unsigned i = 1; i < NetworkInterface::Num; i++) {
    SendBuffer buf;
    // serialize function and data
    gSerialize(buf,lr,f,r,needsReduce);
    //send data
    net.sendLoop (i, &do_all_local_landing_pad<T,FunctionTy>, buf);
  }
  net.handleReceives();
  //Start locally
  do_all_impl(lr,f,r,needsReduce);

  // place a MPI barrier here for all the hosts to synchronize
  //  net.systemBarrier();
}


template<typename IterTy, typename FunctionTy>
FunctionTy do_all_dist(StandardRange<IterTy> r, FunctionTy f, const char* loopname, bool steal) {
  // Get a handle to the network interface
  //  Don't move as NetworkInterface::Num and NetworkInterface::ID have to be initialized first
  NetworkInterface& net = getSystemNetworkInterface();

  typedef typename std::iterator_traits<IterTy>::value_type ItemTy;

  //fast path for non-distributed
  if (NetworkInterface::Num == 1) {
    return do_all_impl(r,f,loopname,steal);
  }

  //copy out all data
  std::deque<ItemTy> allData;
  allData.insert(allData.end(), r.begin(), r.end());

  std::string lname(loopname);

  for (unsigned i = 1; i < NetworkInterface::Num; i++) {
    auto blk = block_range(allData.begin(), allData.end(), i, NetworkInterface::Num);
    std::deque<ItemTy> data(blk.first, blk.second);
    SendBuffer buf;
    // serialize function and data
    gSerialize(buf,f,data, lname, steal);
    //send data
    net.sendLoop (i, &do_all_landing_pad<ItemTy,FunctionTy>, buf);
  }
  net.flush();
  net.handleReceives();
  //now get our data
  auto myblk = block_range(allData.begin(), allData.end(), 0, NetworkInterface::Num);

  //Start locally
  //FIXME: reduce r
  return do_all_impl(Galois::Runtime::makeStandardRange(myblk.first, myblk.second), f, loopname);

  // place a MPI barrier here for all the hosts to synchronize
  // net.systemBarrier();
}

template<typename T, typename FunctionTy>
FunctionTy  do_all_dist(LocalRange<T>& r, FunctionTy f, const char* loopname, bool steal) {
  // Get a handle to the network interface
  //  Don't move as NetworkInterface::Num and NetworkInterface::ID have to be initialized first
  NetworkInterface& net = getSystemNetworkInterface();

  //fast path for non-distributed
  if (NetworkInterface::Num == 1) {
    return do_all_impl(r,f,loopname, steal);
  }

  std::string lname(loopname);

  for (unsigned i = 1; i < NetworkInterface::Num; i++) {
    SendBuffer buf;
    // serialize function and data
    gSerialize(buf,f,r,lname,steal);
    //send data
    net.sendLoop (i, &do_all_local_landing_pad<T,FunctionTy>, buf);
  }
  net.flush();
  net.handleReceives();
  //Start locally
  //FIXME: reduce r
  return do_all_impl(r, f, loopname, steal);

  // place a MPI barrier here for all the hosts to synchronize
  //  net.systemBarrier();
}

template<typename FunctionTy>
void on_each_impl_dist(FunctionTy f, const char* loopname) {
  // Get a handle to the network interface
  //  Don't move as NetworkInterface::Num and NetworkInterface::ID have to be initialized first
  NetworkInterface& net = getSystemNetworkInterface();

  //fast path for non-distributed
  if (NetworkInterface::Num == 1) {
    on_each_impl(f, loopname);
    return;
  }

  for (unsigned i = 1; i < NetworkInterface::Num; i++) {
    SendBuffer buf;
    // serialize function and data
    gSerialize(buf,f);
    //send data
    net.sendLoop (i, &on_each_impl_landing_pad<FunctionTy>, buf);
  }
  net.flush();
  net.handleReceives();
  //Start locally
  on_each_impl(f, loopname);

  // place a MPI barrier here for all the hosts to synchronize
  // net.systemBarrier();
}

struct preAlloc_helper {
  size_t num;

  preAlloc_helper() = default;
  preAlloc_helper(size_t n): num(n) { }

  void operator()(unsigned, unsigned n) {
    int a = n; a = (num + a - 1) / a;
    Galois::Runtime::MM::pagePreAlloc(a); 
  }
};


void preAlloc_impl_dist(int num) {
  on_each_impl_dist(preAlloc_helper(num), nullptr);
}

} // anon
} // Runtime
} // Galois

#endif
