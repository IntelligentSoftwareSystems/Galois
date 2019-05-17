/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

static std::ofstream& openIfNot_send() {
  static std::ofstream MPIsend_file;
  if (!MPIsend_file.is_open()) {
    char name[100] = "";
    gethostname(name, sizeof(name));
    char fname[120];
    snprintf(fname, sizeof(fname), "MPIsend_%s.log", name);
    MPIsend_file.open(fname, std::ios_base::app | std::ofstream::binary);
  }
  assert(MPIsend_file.is_open());
  return MPIsend_file;
}

void galois::runtime::internal::print_send_impl(std::vector<uint8_t> send_vec,
                                                size_t len, unsigned host) {
  using namespace galois::runtime;
  static SimpleLock lock2;
  std::lock_guard<SimpleLock> lg(lock2);
  auto& out = openIfNot_send();
  char buffer[send_vec.size()];
  out << " --> " << host << " Size : " << len << " :  ";
  for (auto x : send_vec) {
    sprintf(buffer, "%u :", x);
    out.write(buffer, sizeof(x));
  }
  out << "\n";
  out.flush();
  // auto& out = openIfNot_send();
  // out.write(reinterpret_cast<const char*>(&send_vec[0]), send_vec.size());
}

void galois::runtime::internal::print_recv_impl(std::vector<uint8_t> recv_vec,
                                                size_t len, unsigned host) {
  using namespace galois::runtime;
  static SimpleLock lock1;
  std::lock_guard<SimpleLock> lg(lock1);
  auto& out = openIfNot_receive();
  char buffer[recv_vec.size()];
  out << host << " <-- "
      << " Size : " << len << " :  ";
  for (auto x : recv_vec) {
    sprintf(buffer, "%u :", x);
    out.write(buffer, sizeof(x));
  }
  out << "\n";
  out.flush();
}

static std::ofstream& openIfNot_receive() {
  static std::ofstream MPIreceive_file;
  if (!MPIreceive_file.is_open()) {
    char name[100] = "";
    gethostname(name, sizeof(name));
    char fname[120];
    snprintf(fname, sizeof(fname), "MPIreceive_%s.log", name);
    MPIreceive_file.open(fname, std::ios_base::app);
  }
  assert(MPIreceive_file.is_open());
  return MPIreceive_file;
}
