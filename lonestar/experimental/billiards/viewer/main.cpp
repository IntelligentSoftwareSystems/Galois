/** Simple OpenGL viewer for billiards simulation -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Viewer.h"
#include <qapplication.h>
#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;

static cll::opt<unsigned> refdelay("refdelay", cll::desc("Delay between frame refreshes (ms)"), cll::init(100));
static cll::opt<std::string> configFilename(cll::Positional, cll::desc("<input file>"), cll::init("config.csv"));
static cll::opt<std::string> eventLogFilename(cll::Positional, cll::desc("<event log file>"), cll::init("simLog.csv"));

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  QApplication application(argc, argv);
  Scene scene(configFilename, eventLogFilename);
  Viewer viewer(scene, refdelay);

#if QT_VERSION < 0x040000
  application.setMainWidget(&viewer);
#else
  viewer.setWindowTitle("Billiards Demo");
#endif

  viewer.show();

  return application.exec();
}
