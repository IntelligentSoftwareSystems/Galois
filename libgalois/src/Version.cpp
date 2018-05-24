#include "galois/Version.h"

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

std::string galois::getVersion() {
  return STR(GALOIS_VERSION);
}

std::string galois::getRevision() {
  return "unknown";
}

int galois::getVersionMajor() {
  return GALOIS_VERSION_MAJOR;
}

int galois::getVersionMinor() {
  return GALOIS_VERSION_MINOR;
}

int galois::getVersionPatch() {
  return GALOIS_VERSION_PATCH;
}

int galois::getCopyrightYear() {
  return GALOIS_COPYRIGHT_YEAR;
}
