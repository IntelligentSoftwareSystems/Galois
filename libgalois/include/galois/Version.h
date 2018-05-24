#ifndef GALOIS_VERSION_H
#define GALOIS_VERSION_H

#include <string>

namespace galois {

std::string getVersion();
std::string getRevision();
int getVersionMajor();
int getVersionMinor();
int getVersionPatch();
int getCopyrightYear();

} // end namespace galois

#endif
