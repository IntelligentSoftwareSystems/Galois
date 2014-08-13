#include "Galois/Runtime/TreeExec.h"

namespace Galois {
namespace Runtime {

TreeExecGeneric* treeExecPtr = nullptr;

TreeExecGeneric& getTreeExecutor (void) {
  return *treeExecPtr;
}

void setTreeExecutor (TreeExecGeneric* t) {
  treeExecPtr = t;
}

void spawn (std::function<void (void)> f) {
  getTreeExecutor ().push (f);
}

void sync (void) {
  getTreeExecutor ().syncLoop ();
}

void for_each_ordered_tree_generic (std::function<void (void)> initTask, const char* loopname) {

  TreeExecGeneric e (loopname);

  e.push (initTask);

  setTreeExecutor (&e);

  getSystemThreadPool ().run (Galois::getActiveThreads (),
      [&e] (void) { e.initThread (); },
      std::ref (e));


  setTreeExecutor (nullptr);
}

} // end namespace Runtime
} // end namespace Galois
