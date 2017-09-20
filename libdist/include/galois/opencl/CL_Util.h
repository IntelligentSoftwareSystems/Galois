/**
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Rashid Kaleem<rashid.kaleem@gmail.com>
 */
#ifdef __APPLE__
#include <opencl/opencl.h>
#else
extern "C" {
#include "CL/cl.h"
}
;
#endif

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/stat.h>
#include <assert.h>
#include <string>
#include "galois/opencl/CL_Errors.h"
#include "galois/opencl/CL_DeviceManager.h"
#include "galois/opencl/Arrays/Arrays.h"
//////////////////////////////////////////
#ifndef GOPT_CL_UTIL_H_
#define GOPT_CL_UTIL_H_
namespace galois {
namespace opencl {


///////////////////////////////////////////////////////////////////////////////////
template<typename EType, typename NType>
class DIMACS_GR_Challenge9_Format {
public:
   static bool is_comment(std::string & str) {
      return str.c_str()[0] == 'c';
   }
   static std::pair<size_t, size_t> parse_header(std::string & s) {
      if (s.c_str()[0] == 'p') {
         char buff[256];
         strcpy(buff, s.substr(1, s.size()).c_str());
         char * tok = strtok(buff, " ");
         tok = strtok(NULL, " "); // Ignore problem name for challenge9 formats.
         size_t num_nodes = atoi(tok) - 1; // edges start from zero.
         tok = strtok(NULL, " ");
         size_t num_edges = atoi(tok);
         return std::pair<size_t, size_t>(num_nodes, num_edges);
      }
      return std::pair<size_t, size_t>((size_t) -1, (size_t) -1);
   }
   static std::pair<NType, std::pair<NType, EType> > parse_edge_pair(std::string & s) {
      if (s.c_str()[0] == 'a') {
         char buff[256];
         strcpy(buff, s.substr(1, s.size()).c_str());
         char * tok = strtok(buff, " ");
         NType src_node = atoi(tok) - 1;
         tok = strtok(NULL, " ");
         NType dst_node = atoi(tok) - 1;
         tok = strtok(NULL, " ");
         EType edge_wt = atoi(tok);
         return std::pair<NType, std::pair<NType, EType> >(src_node, std::pair<NType, EType>(dst_node, edge_wt));
      }
      return std::pair<NType, std::pair<NType, EType> >(-1, std::pair<NType, EType>(-1, -1));
   }
};
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
////// Generate a random float between 'low' and 'high'
/////////////////////////////////////////////////////////////////////////////////////
inline float rrange(float low, float high) {
   return (int) rand() * (high - low) / (float) RAND_MAX + low;
}
#ifdef _WIN32
inline double time_event_seconds(cl_event & event ) {
#else
inline double time_event_seconds(cl_event & event __attribute__((unused))) {
#endif

#ifdef _CL_PROFILE_EVENTS_
   cl_ulong start_time, end_time;
   clWaitForEvents(1, &event);
   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
   clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
   double total_time = ((end_time - start_time)/((double)(1000*1000*1000)));
   return total_time;
#else
   return 0;
#endif
}
/////////////////////////////
inline int next_power_2(int x) {
   if (x < 0)
      return 0;
   --x;
   x |= x >> 1;
   x |= x >> 2;
   x |= x >> 4;
   x |= x >> 8;
   x |= x >> 16;
   return x + 1;
}
/////////////////////////////
}
}  //End namespaces

#endif /* GOPT_CL_UTIL_H_ */
