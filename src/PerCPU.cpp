#include <vector>
#include <algorithm>
#include <iostream>

#include "Galois/Runtime/PerCPU.h"
#include "Support/ThreadSafe/simple_lock.h"

//This, once initialized by a thread, stores an dense index/label for that thread
static __thread int LocalThreadID = 0;

//This stores the next thread id
static int nextThreadID = 0;


int GaloisRuntime::getThreadID() {
  int ID = LocalThreadID;
  if (!ID) {
    //Not previously accessed
    ID = __sync_add_and_fetch(&nextThreadID, 1);
    LocalThreadID = ID;
  }
  assert(ID > 0 && ID < 64);
  return ID;
}

int GaloisRuntime::getMaxThreadID() {
  return nextThreadID;
}


union LocalDataItem {
  long longData;
  int intData;
  short shortData;
  char charData;
  bool boolData;
  void* ptrData;
};

static __thread LocalDataItem* ThreadLocalData = 0;
static __thread int ThreadLocalNum = 0;

//This is the total number allocated so far
static int nextLocalItem = 0;

static std::vector<std::pair<LocalDataItem*, int> > AllThreadData;
static threadsafe::simpleLock<int,true> AllThreadDataLock;

static LocalDataItem setToZero() {
  LocalDataItem LDI;
  LDI.longData = 0;
  LDI.intData = 0;
  LDI.shortData = 0;
  LDI.charData = 0;
  LDI.boolData = 0;
  LDI.ptrData = 0;
  return LDI;
}

static LocalDataItem* createAll() {
  //We only create inside a thread, so no concurrency worry
  int num = nextLocalItem;
  LocalDataItem* L = new LocalDataItem[num];
  std::generate(&L[0], &L[num], setToZero);
  ThreadLocalData = L;
  ThreadLocalNum = num;
  AllThreadDataLock.lock();
  int myID = GaloisRuntime::getThreadID();
  AllThreadData.resize(std::max(1 + myID, (int)AllThreadData.size()));
  AllThreadData[myID].first = L;
  AllThreadData[myID].second = num;
  AllThreadDataLock.unlock();
  return L;
}

static LocalDataItem* getIndexedChecked(int index) {
  LocalDataItem* L = ThreadLocalData;
  assert(index < nextLocalItem); //Check global safety
  if (!L) {
    L = createAll();
  } else if (ThreadLocalNum <= index) {
    LocalDataItem* Old = L;
    long oldsize = ThreadLocalNum;
    L = createAll();
    std::copy(&Old[0], &Old[oldsize], &L[0]);
    std::cerr << "Copy Happened\n";
    delete Old;
  }
  return &L[index];
}

static int createNewID() {
  return __sync_fetch_and_add(&nextLocalItem, 1);
}

template<typename T> T* getByType(LocalDataItem* I);
template<> bool*  getByType(LocalDataItem* I) { return &(I->boolData);}
template<> char*  getByType(LocalDataItem* I) { return &(I->charData);}
template<> short* getByType(LocalDataItem* I) { return &(I->shortData);}
template<> int*   getByType(LocalDataItem* I) { return &(I->intData);}
template<> long*  getByType(LocalDataItem* I) { return &(I->longData);}
template<> void** getByType(LocalDataItem* I) { return &(I->ptrData);}


namespace GaloisRuntime {

  template<typename T>
  PerCPUData<T>::PerCPUData() {
    index = createNewID();
  }
  
  template<typename T>
  PerCPUData<T>::~PerCPUData() {
  }
  
  template<typename T>
  T PerCPUData<T>::get() const {
    return *(getByType<T>(getIndexedChecked(index)));
  }

  template<typename T>
  void PerCPUData<T>::set(const T& d) {
    *(getByType<T>(getIndexedChecked(index))) = d;
  }

  template<typename T>
  void PerCPUData<T>::set_assert(const T& d, const T& old) {
    T* V = getByType<T>(getIndexedChecked(index));
    assert(*V == old);
    *V = d;
  }
  

  template<typename T>
  T PerCPUData<T>::getRemote(int rThreadID) const {
    T retval = (T)0;
    AllThreadDataLock.lock();
    AllThreadData.resize(std::max(1 + rThreadID, (int)AllThreadData.size()));
    LocalDataItem* L = AllThreadData[rThreadID].first;
    int num = AllThreadData[rThreadID].second;
    if (L && (index < num))
      retval = *(getByType<T>(&L[index]));
    AllThreadDataLock.unlock();
    return retval;
  }

  //Instantiate for supported types
  template class PerCPUData<bool>;
  template class PerCPUData<char>;
  template class PerCPUData<short>;
  template class PerCPUData<int>;
  template class PerCPUData<long>;
  template class PerCPUData<void*>;
}
