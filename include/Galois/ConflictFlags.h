// Galois Conflict flags -*- C++ -*-
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

#ifndef GALOIS_CONFLICTFLAGS_H
#define GALOIS_CONFLICTFLAGS_H

namespace Galois {

//! What should the runtime do when executing a method.
/*! Various methods take an optional parameter indicating what actions the runtime should do on the user's behalf: (1) checking for conflicts, and/or (2) saving undo information. By default, both are performed (ALL).
 */
enum MethodFlag {
  NONE, ALL, CHECK_CONFLICT, SAVE_UNDO
};

}

#endif //GALOIS_CONFLICTFLAGS_H
