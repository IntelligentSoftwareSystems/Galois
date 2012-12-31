#include "Galois/Runtime/LoopHooks.h"

#include "Galois/Runtime/ll/TID.h"

using namespace Galois::Runtime;

static AtLoopExit* head;

AtLoopExit::AtLoopExit() {
  next = head;
  head = this;
}

AtLoopExit::~AtLoopExit() {
  if (head == this) {
    head = next;
  } else {
    AtLoopExit* prev = head;
    while (prev->next != this) { prev = prev->next; }
    prev->next = next;
  }
}

void Galois::Runtime::runAllLoopExitHandlers() {
  if (LL::getTID() == 0) {
    AtLoopExit* h = head;
    while (h) {
      h->LoopExit();
      h = h->next;
    }
  }
}
