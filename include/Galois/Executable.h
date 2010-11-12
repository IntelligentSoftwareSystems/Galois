// -*- C++ -*-
/*! \file 
 *  \brief simple Executable base class
 */

#ifndef _GALOIS_EXECUTABLE_H
#define _GALOIS_EXECUTABLE_H

namespace Galois {
  
class Executable {
public:
  //! run work.  id is the thread id, max is the total num
  virtual void operator()(int id, int tmax) = 0;
  
  //! execute before work is run to let any local, number of threads
  //! dependent variables get initialized
  virtual void preRun(int tmax) {}
  
  //! execute after work is run to perform any cleanup
  virtual void postRun() {}
};

}

#endif
