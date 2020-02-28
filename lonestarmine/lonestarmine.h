#pragma once
#include <string>
#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: unsymmetrized graph>"), cll::Required);
static cll::opt<unsigned> num_trials("n", cll::desc("perform n trials (default value 1)"), cll::init(1));
static cll::opt<unsigned> num_blocks("b", cll::desc("edge blocking to b blocks (default value 1)"), cll::init(1));
#ifdef USE_QUERY_GRAPH
static cll::opt<std::string> query_graph_filename(cll::Positional, cll::desc("<filename: unsymmetrized graph>"), cll::Required);
static cll::opt<std::string> preset_filename("pf", cll::desc("<filename: preset matching order>"), cll::init("default"));
static cll::opt<unsigned> fv("fv", cll::desc("first vertex is special"), cll::init(0));
#endif
#ifndef TRIANGLE
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-clique (default value 3)"), cll::init(3));
#endif
static cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
static cll::opt<unsigned> debug("d", cll::desc("print out the frequent patterns for debugging"), cll::init(0));
#ifdef EDGE_INDUCED
static cll::opt<unsigned> minsup("ms", cll::desc("minimum support (default value 0)"), cll::init(0));
#endif
 
void LonestarMineStart(int argc, char** argv, const char* app, const char* desc, const char* url) {
	//llvm::cl::SetVersionPrinter(LonestarPrintVersion);
	llvm::cl::ParseCommandLineOptions(argc, argv);
	std::cout << "Copyright (C) 2020 The University of Texas at Austin\n";
	std::cout << "http://iss.ices.utexas.edu/galois/\n\n";
	std::cout << "application: " << (app ? app : "unspecified") << "\n";
	if (desc) std::cout << desc << "\n";
	if (url) std::cout << "http://iss.ices.utexas.edu/?p=projects/galois/benchmarks/" << url << "\n";
	std::cout << "\n";
}

