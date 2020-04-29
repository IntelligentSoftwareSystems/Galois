#include "galois/OutIndexView.h"
#include "tsuba/tsuba.h"

namespace galois {

int OutIndexView::Bind() {
  struct GRHeader header;
  int err;
  if (err = TsubaPeek(filename_, &header); err) {
    perror(filename_.c_str());
    return err;
  }
  err = file_.Bind(filename_,
                   sizeof(header) + header.num_nodes_ * sizeof(index_t));
  if (err) {
    return err;
  }
  gr_view_ = file_.ptr<GRPrefix>();
  return 0;
}

void OutIndexView::Unbind() {
  file_.Unbind();
  gr_view_ = nullptr;
}

} /* namespace galois */
