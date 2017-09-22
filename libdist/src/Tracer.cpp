/** Galois Distributed Object Tracer -*- C++ -*-
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

#include "galois/runtime/Tracer.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/EnvCheck.h"

#include <fstream>
#include <cassert>
#include <iostream>
#include <chrono>
#include <mutex>

#include <sys/types.h>
#include <unistd.h>

using namespace galois::substrate;

static bool doCerr = false;
static bool doCerrInit = false;

namespace galois { 
namespace runtime {
uint32_t getHostID() __attribute__((weak));
} // end namespace runtime
} // end namespace galois

uint32_t galois::runtime::getHostID() {
  return 0;
}

static std::ostream& openIfNot() {
  if (!doCerrInit) {
    doCerr = EnvCheck("GALOIS_DEBUG_TRACE_STDERR");
    doCerrInit = true;
  }
  if (doCerr)
    return std::cerr;
  static std::ofstream output;
  if (!output.is_open()) {
    pid_t id = getpid();
    char name[100] = "";
    gethostname(name, sizeof(name));
    char fname[120];
    snprintf(fname, sizeof(fname), "%s.%d.log", name, id);
    output.open(fname, std::ios_base::app);
  }
  assert(output.is_open());
  return output;
}

void galois::runtime::internal::printTrace(std::ostringstream& os) {
  using namespace std::chrono;
  static SimpleLock lock;
  std::lock_guard<SimpleLock> lg(lock);
  auto& out = openIfNot();
  auto dtn = system_clock::now().time_since_epoch();
  out << "<" << dtn.count() << "," << getHostID() << "> ";
  out << os.str();
  out.flush();
  static int iSleep = 0;
  static bool doSleep = EnvCheck("GALOIS_DEBUG_TRACE_PAUSE", iSleep);
  if (doSleep)
    usleep(iSleep ? iSleep : 10);
}



static std::ofstream& openIfNot_receive() {
  static std::ofstream MPIreceive_file;
  if(!MPIreceive_file.is_open()){
    char name[100] = "";
    gethostname(name, sizeof(name));
    char fname[120];
    snprintf(fname, sizeof(fname), "MPIreceive_%s.log", name);
    MPIreceive_file.open(fname, std::ios_base::app);
  }
  assert(MPIreceive_file.is_open());
  return MPIreceive_file;
}
static std::ofstream& openIfNot_send() {
  static std::ofstream MPIsend_file;
  if(!MPIsend_file.is_open()){
    char name[100] = "";
    gethostname(name, sizeof(name));
    char fname[120];
    snprintf(fname, sizeof(fname), "MPIsend_%s.log", name);
    MPIsend_file.open(fname, std::ios_base::app | std::ofstream::binary);
  }
  assert(MPIsend_file.is_open());
  return MPIsend_file;
}

static std::ofstream& openIfNot_output() {
  static std::ofstream output_file;
  if(!output_file.is_open()){
    char name[100] = "";
    gethostname(name, sizeof(name));
    char fname[120];
    snprintf(fname, sizeof(fname), "output_%s_%d.log", name, 
             galois::runtime::getHostID());
    output_file.open(fname, std::ios_base::app);
  }
  assert(output_file.is_open());
  return output_file;
}


void galois::runtime::internal::print_recv_impl(std::vector<uint8_t> recv_vec, 
                                                size_t len, unsigned host) {
  using namespace galois::runtime;
  static SimpleLock lock1;
  std::lock_guard<SimpleLock> lg(lock1);
  auto& out = openIfNot_receive();
  char buffer[recv_vec.size()];
  out << host <<" <-- " << " Size : " << len << " :  ";
  for(auto x : recv_vec){
    sprintf(buffer, "%u :", x);
    out.write(buffer, sizeof(x));
  }
  out << "\n";
  out.flush();
}
void galois::runtime::internal::print_send_impl(std::vector<uint8_t> send_vec, 
                                                size_t len, unsigned host){
  using namespace galois::runtime;
  static SimpleLock lock2;
  std::lock_guard<SimpleLock> lg(lock2);
  auto& out = openIfNot_send();
  char buffer[send_vec.size()];
  out << " --> " << host << " Size : " << len << " :  ";
  for(auto x : send_vec){
    sprintf(buffer, "%u :", x);
    out.write(buffer, sizeof(x));
  }
  out << "\n";
  out.flush();
  //auto& out = openIfNot_send();
  //out.write(reinterpret_cast<const char*>(&send_vec[0]), send_vec.size());
}


void galois::runtime::internal::print_output_impl(std::ostringstream& os){
  using namespace galois::runtime;
  static SimpleLock lock2;
  std::lock_guard<SimpleLock> lg(lock2);
  auto& out = openIfNot_output();
  out << os.str();
  out.flush();
}

bool galois::runtime::internal::doTrace = false;
bool galois::runtime::internal::initTrace = false;
