// -*- C++ -*-

void printBanner(std::ostream& OS, const char* app, const char* desc, const char* url) {
  OS << "\nLonestar Benchmark Suite v3.0 (C++)\n"
       << "Copyright (C) 2007, 2008, 2009, 2010 The University of Texas at Austin\n"
       << "http://iss.ices.utexas.edu/lonestar/\n"
       << "\n"
       << "application: " << app << "\n";
  if (desc)
    OS << desc << "\n";
  if (url)
    OS << url << "\n";
  OS << "\n";
}
