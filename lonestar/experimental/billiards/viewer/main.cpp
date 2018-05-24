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
