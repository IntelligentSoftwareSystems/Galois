// Uniform Reporting mechanism -*- C++ -*-

#ifndef __GALOIS_SUPPORT_H_
#define __GALOIS_SUPPORT_H_

namespace GaloisRuntime {

//Report Statistics
void reportStat(const char* text, unsigned long val);
void reportStat(const char* text, unsigned int val);
void reportStat(const char* text, double val);
void reportStat(const char* text, const char* val);

//Report Warnings
void reportWarning(const char* text);
void reportWarning(const char* text, unsigned int val);
void reportWarning(const char* text, unsigned long val);
void reportWarning(const char* text, const char* val);

}

#endif

