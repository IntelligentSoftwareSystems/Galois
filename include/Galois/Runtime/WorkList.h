// Scalable Chunked worklist -*- C++ -*-

#include "Support/ThreadSafe/simple_lock.h"
#include <queue>
#include <stack>

namespace Galois {
  template<typename T>
  class WorkList;
}

namespace GaloisRuntime {

  template<typename T>
  class GWL_LIFO : public Galois::WorkList<T> {
    std::stack<T> wl;
    threadsafe::simpleLock lock;

  public:
    //These should only be called by one thread
    virtual void push(T val) {
      lock.write_lock();
      wl.push(val);
      lock.write_unlock();
    }

    T pop(bool& succeeded) {
      lock.write_lock();
      if (wl.empty()) {
	succeeded = false;
	lock.write_unlock();
	return T();
      } else {
	succeeded = true;
	T retval = wl.top();
	wl.pop();
	lock.write_unlock();
	return retval;
      }
    }
    
    //This can be called by any thread
    T steal(bool& succeeded) {
      return pop(succeeded);
    }

    bool empty() {
      lock.write_lock();
      bool retval = wl.empty();
      lock.write_unlock();
      return retval;
    }
  };


  template<typename T>
  class GWL_FIFO : public Galois::WorkList<T> {
    std::queue<T> wl;
    threadsafe::simpleLock lock;

  public:
    //These should only be called by one thread
    virtual void push(T val) {
      lock.write_lock();
      wl.push(val);
      lock.write_unlock();
    }

    T pop(bool& succeeded) {
      lock.write_lock();
      if (wl.empty()) {
	succeeded = false;
	lock.write_unlock();
	return T();
      } else {
	succeeded = true;
	T retval = wl.top();
	wl.pop();
	lock.write_unlock();
	return retval;
      }
    }
    
    //This can be called by any thread
    T steal(bool& succeeded) {
      return pop(succeeded);
    }

    bool empty() {
      lock.write_lock();
      bool retval = wl.empty();
      lock.write_unlock();
      return retval;
    }
  };

}
