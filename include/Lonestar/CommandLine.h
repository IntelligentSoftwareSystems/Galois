
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
