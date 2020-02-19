#pragma once

#include <iostream>
#include <sstream>
#include "galois/Galois.h"
#include "galois/Version.h"
#include "llvm/Support/CommandLine.h"

//! standard global options to the benchmarks
extern llvm::cl::opt<bool> skipVerify;
extern llvm::cl::opt<int> numThreads;
extern llvm::cl::opt<std::string> statFile;

//! standard global options to the benchmarks
llvm::cl::opt<bool> skipVerify("noverify", llvm::cl::desc("Skip verification step (default value false)"), llvm::cl::init(false));
llvm::cl::opt<int>numThreads("t", llvm::cl::desc("Number of threads (default value 1)"), llvm::cl::init(1));
llvm::cl::opt<std::string> statFile("statFile", llvm::cl::desc("ouput file to print stats to (default value empty)"), llvm::cl::init(""));

static void LonestarGnnPrintVersion() {
	std::cout << "LoneStar Benchmark Suite v" << galois::getVersion() << " (" << galois::getRevision() << ")\n";
}

//! initialize lonestargnn benchmark
void LonestarGnnStart(int argc, char** argv, const char* app, const char* desc, const char* url) {
	llvm::cl::SetVersionPrinter(LonestarGnnPrintVersion);
	llvm::cl::ParseCommandLineOptions(argc, argv);
	numThreads = galois::setActiveThreads(numThreads);
	galois::runtime::setStatFile(statFile);
	LonestarGnnPrintVersion();
	std::cout << "Copyright (C) " << galois::getCopyrightYear() << " The University of Texas at Austin\n";
	std::cout << "http://iss.ices.utexas.edu/galois/\n\n";
	std::cout << "application: " << (app ? app : "unspecified") << "\n";
	if (desc) std::cout << desc << "\n";
	if (url) std::cout << "http://iss.ices.utexas.edu/?p=projects/galois/benchmarks/" << url << "\n";
	std::cout << "\n";

	std::ostringstream cmdout;
	for (int i = 0; i < argc; ++i) {
		cmdout << argv[i];
		if (i != argc - 1) cmdout << " ";
	}

	galois::runtime::reportParam("(NULL)", "CommandLine", cmdout.str());
	galois::runtime::reportParam("(NULL)", "Threads", numThreads);

	char name[256];
	gethostname(name, 256);
	galois::runtime::reportParam("(NULL)", "Hostname", name);
}

