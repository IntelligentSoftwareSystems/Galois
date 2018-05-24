#include "galois/runtime/TreeExec.h"

namespace galois {
namespace runtime {

void for_each_ordered_tree_generic (TreeTaskBase& initTask, const char* loopname) {
  for_each_ordered_tree_impl<TreeTaskBase> (initTask, loopname);
}

} // end namespace runtime
} // end namespace galois
