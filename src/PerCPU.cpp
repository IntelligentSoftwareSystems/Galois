#include <vector>
#include <string.h>
#include <algorithm>

#include "Galois/Runtime/PerCPU.h"
#include "Support/ThreadSafe/simple_lock.h"

//This, once initialized by a thread, stores an dense index/label for that thread
static __thread int ThreadID = 0;

//This stores the next thread id
static int nextThreadID = 0;


int GaloisRuntime::getThreadID() {
  int ID = ThreadID;
  if (!ID) {
    //Not previously accessed
    ID = __sync_add_and_fetch(&nextThreadID, 1);
    ThreadID = ID;
  }
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

//index 0 always stores the local length of the array
static __thread LocalDataItem* ThreadLocalData = 0;
//This is the total number allocated so far
//a full array is this + 1 (local length field)
static int nextLocalItem = 0;

static std::vector<LocalDataItem*> AllThreadData;
static threadsafe::simpleLock AllThreadDataLock;

//returned indexes start at zero and index into i + 1 of the array

static __attribute__((noinline)) LocalDataItem* getIndexedRemote(int index, LocalDataItem* L) {
  if (L[0].longData <= index) {
    return 0;
  }
  return &L[index + 1];
}

static LocalDataItem* getIndexedChecked(int index) {
  LocalDataItem* L = ThreadLocalData;
  assert(index < nextLocalItem); //Check global safety
  if (!L) {
    AllThreadDataLock.write_lock();
    //We only create inside a thread, so no concurrency worry
    int num = nextLocalItem;
    ThreadLocalData = L = new LocalDataItem[num + 1];
    L[0].longData = num;
    AllThreadData.resize(std::max(1 + GaloisRuntime::getThreadID(), (int)AllThreadData.size()));
    AllThreadData[GaloisRuntime::getThreadID()] = L;
    AllThreadDataLock.write_unlock();
  } else if (L[0].longData <= index) {
    AllThreadDataLock.write_lock();
    LocalDataItem* Old = L;
    long oldsize = L[0].longData;
    int num = nextLocalItem;
    ThreadLocalData = L = new LocalDataItem[num + 1];
    bzero(L, sizeof(LocalDataItem[num + 1]));
    memcpy(L, Old, sizeof(LocalDataItem[num + 1]));
    L[0].longData = num;
    delete Old;
    AllThreadData.resize(std::max(1 + GaloisRuntime::getThreadID(), (int)AllThreadData.size()));
    AllThreadData[GaloisRuntime::getThreadID()] = L;
    AllThreadDataLock.write_unlock();    
  }
  return &L[index + 1];
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
    AllThreadDataLock.write_lock();
    AllThreadData.resize(std::max(1 + rThreadID, (int)AllThreadData.size()));
    if (LocalDataItem* L = AllThreadData[rThreadID]) {
      L = getIndexedRemote(index, L);
      if (L)
	retval = *(getByType<T>(L));
    }
    AllThreadDataLock.write_unlock();
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
