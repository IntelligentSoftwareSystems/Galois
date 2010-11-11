// simple non-blocking spin lock for cray-*- C++ -*-

#ifndef _SIMPLE_LOCK_H
#define _SIMPLE_LOCK_H

#include <cassert>

namespace threadsafe {
namespace cray {

// The most stupid spinlock you can imagine
class simpleLock {
  int _lock;
public:
  simpleLock() {
    writexf(&_lock, 0); // sets to full
  }
  void read_lock() {
    while (!try_read_lock()) {}
  }

  void read_unlock() {
    readfe(&_lock); // sets to empty, acquiring the lock lock
    writeef(&_lock, 0); // clears the lock and clears the lock lock
  }

  bool try_read_lock() {
    int V = readfe(&_lock); // sets to empty, acquiring the lock lock
    if (V) {
      //failed
      writeef(&_lock, V); //write back the same value and clear the lock lock
      return false;
    } else {
      //can grab
      writeef(&_lock, 1); //write our value into the lock (acquire) and clear the lock lock
      return true;
    }
  }

  void promote() {}

  void write_lock() {
    read_lock();
  }
  void write_unlock() {
    read_unlock();
  }
  bool try_write_lock() {
    return try_read_lock();
  }
};

class ptrLock {
  void* _lock;
public:

  ptrLock() {
    writexf(&_lock, 0); // sets to full
  }

  void lock(void* val) {
    //      assert(!_lock);
    while (!try_lock(val)) {}
  }
  void unlock() {
    readfe(&_lock); // sets to empty, acquiring the lock lock
    writeef(&_lock, 0); // clears the lock and clears the lock lock
  }
  bool try_lock(void* val) {
    void* V = readfe(&_lock); // sets to empty, acquiring the lock lock
    if (V && V != val) {
      //failed
      writeef(&_lock, V); //write back the same value and clear the lock lock
      return false;
    } else {
      //can grab
      writeef(&_lock, val); //write our value into the lock (acquire) and clear the lock lock
      return true;
    }
  }
  void* getValue() {
    return readxx(_lock); //does not touch the lock lock
  }
};

}
}

#endif
