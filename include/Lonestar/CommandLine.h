/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

static bool skipVerify = false;
static long numThreads = 1;


//pulls out common options and returns the rest
std::vector<const char*> parse_command_line(int argc, const char** argv, const char* proghelp) {
  std::vector<const char*> retval;

  //known options
  //-t threads
  //-noverify
  //-help

  for (int i = 1; i < argc; ++i) {
    if (std::string("-t").compare(argv[i]) == 0) {
      if (i + 1 >= argc) {
	std::cerr << "Error parsing -t option, missing number\n";
	abort();
      }
      char* endptr = 0;
      numThreads = strtol(argv[i+1], &endptr, 10);
      if (endptr == argv[i+1]) {
	std::cerr << "Error parsing -t option, number not recognized\n";
	abort();
      }
      Galois::setMaxThreads(numThreads);
      ++i; //eat arg value
    } else if (std::string("-noverify").compare(argv[i]) == 0) {
      skipVerify = true;
    } else if (std::string("-help").compare(argv[i]) == 0) {
      std::cout << "[-t numThreads] use numThreads threads (1)\n"
		<< "[-noverify] skip verification\n"
		<< proghelp << "\n";
    } else {
      retval.push_back(argv[i]);
    }
  }
  return retval;
}
