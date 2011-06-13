// Uniform Reporting mechanism -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

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

//Report Info
void reportInfo(const char* text);
void reportInfo(const char* text, unsigned int val);
void reportInfo(const char* text, unsigned long val);
void reportInfo(const char* text, const char* val);

}

#endif

