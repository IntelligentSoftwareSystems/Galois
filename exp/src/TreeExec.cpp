#include "Galois/Runtime/TreeExec.h"

namespace Galois {
namespace Runtime {

void for_each_ordered_tree_generic (TreeTaskBase& initTask, const char* loopname) {
  for_each_ordered_tree_impl<TreeTaskBase> (initTask, loopname);
}

} // end namespace Runtime
} // end namespace Galois
