#ifndef GALOIS_CONFIG_H
#define GALOIS_CONFIG_H

#include <galois/graphs/Graph.h>

#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>

class Config {
public:
  double tolerance;
  bool version2D;
  int steps;
  int cores;
  bool display;

  // Lat/lon coordinates of the bounding box of the domain
  // These values will not be used when a mesh file is given
  double N;
  double S;
  double E;
  double W;

  std::string dataDir; // Directory of the DTM maps and mesh files (if used)

  bool ascii;
  std::string asciiFile;

  std::string inputMeshFile; // Filename of the inp mesh. It should be inside dataDir and use UTM coordinates
  long zone;                 // UTM zone of the inputMeshFile
  char hemisphere;           // Hemisphere of the inputMeshFile

  std::string output;

  Config(int argc, char** argv)
      : tolerance(5), version2D(false), steps(11), cores(-1), display(false),
        N(50.2), S(49.9), E(20.2), W(19.7), dataDir("data"), ascii(false),
        asciiFile(""), inputMeshFile(""), zone(34), hemisphere('N'),
        output(std::string("graph") +
               std::to_string(galois::getActiveThreads()) + ".mgf") {
    parseArguments(argc, argv);
  }

  static Config getConfig(int argc, char** argv) { Config config{argc, argv}; }

private:
  inline static const std::string USAGE =
      "OPTIONS:\n"
      //                                            "\t-t <tolerance>\n"
      //                                            "\t-s <steps>\n"
      //                                            "\t-2 turns on 2D version\n"
      //                                            "\t-3 turns on 3D version\n"
      "\n";

  void parseArguments(int argc, char** argv) {
    int argument;
    if (argc == 1) {
      fprintf(stderr, "%s", USAGE.c_str());
    }
    while ((argument = getopt(argc, argv, "T:23s:dN:S:E:W:D:af:o:c:")) != -1)
      switch (argument) {
      case 'T':
        tolerance = strtod(optarg, nullptr);
        break;
      case '2':
        version2D = true;
        break;
      case '3':
        version2D = false;
        break;
      case 's':
        steps = strtoimax(optarg, nullptr, 10);
        break;
      case 'd':
        display = true;
        break;
      case 'N':
        N = strtod(optarg, nullptr);
        break;
      case 'S':
        S = strtod(optarg, nullptr);
        break;
      case 'E':
        E = strtod(optarg, nullptr);
        break;
      case 'W':
        W = strtod(optarg, nullptr);
        break;
      case 'D':
        dataDir = optarg;
        break;
      case 'a':
        ascii = true;
        break;
      case 'f':
        asciiFile = optarg;
        break;
      case 'o':
        output = optarg;
        break;
      case 'c':
        cores = strtoimax(optarg, nullptr, 10);
        break;
      case '?':
        if (optopt == 'T' || optopt == 's' || optopt == '2' || optopt == '3' ||
            optopt == 'd' || optopt == 'N' || optopt == 'S' || optopt == 'E' ||
            optopt == 'W' || optopt == 'D' || optopt == 'a' || optopt == 'f' ||
            optopt == 'o' || optopt == 'c') {

          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
          fprintf(stderr, "%s", USAGE.c_str());
        } else if (isprint(optopt)) {
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
          fprintf(stderr, "%s", USAGE.c_str());
        } else {
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
          fprintf(stderr, "%s", USAGE.c_str());
        }
        //                    exit(1);
      default:
        break;
      }
  }
};

#endif // GALOIS_CONFIG_H
