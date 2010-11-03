// Scalable Chunked worklist -*- C++ -*-

#include "Support/ThreadSafe/simple_lock.h"
#include "Support/PackedInt.h"
#include <queue>
#include <stack>

namespace Galois {
  template<typename T>
  class WorkList;
}

#define NOCOPY(_name)				\
  _name(const _name& rhs);			\
  _name& operator=(const _name&)

namespace GaloisRuntime {

  template<typename T>
  class GWL_LIFO : public Galois::WorkList<T> {
    std::stack<T> wl;
    threadsafe::simpleLock lock;

    //Don't allow copying
    NOCOPY(GWL_LIFO);
    
  public:

    GWL_LIFO() {}
    ~GWL_LIFO() {}

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
  class GWL_LIFO_SB : public Galois::WorkList<T> {
    std::deque<T> wl;
    threadsafe::simpleLock lock;

    NOCOPY(GWL_LIFO_SB);

  public:

    GWL_LIFO_SB() {}
    ~GWL_LIFO_SB() {}

    //These should only be called by one thread
    virtual void push(T val) {
      lock.write_lock();
      wl.push_back(val);
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
	T retval = wl.back();
	wl.pop_back();
	lock.write_unlock();
	return retval;
      }
    }
    
    //This can be called by any thread
    T steal(bool& succeeded) {
      lock.write_lock();
      if (wl.empty()) {
	succeeded = false;
	lock.write_unlock();
	return T();
      } else {
	succeeded = true;
	T retval = wl.front();
	wl.pop_front();
	lock.write_unlock();
	return retval;
      }

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

    NOCOPY(GWL_FIFO);

  public:

    GWL_FIFO() {}
    ~GWL_FIFO() {}

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

  //This is buggy
  template<typename T>  
  class GWL_ChaseLev_Dyn : public Galois::WorkList<T> {

    NOCOPY(GWL_ChaseLev_Dyn);

    struct DequeNode {
      enum { ArraySize = 256 };
	T itsDataArr[ArraySize];
      DequeNode* next;
      DequeNode* prev;
    };

    // struct BottomStruct {
    //   DequeNode* nodeP;
    //   int32_t cellIndex;
    // };

    // struct TopStruct {
    //   DequeNode* nodeP;
    //   int32_t cellIndex;
    //   int32_t tag;
    // };

    //FIXME: generalize for 32 bit hosts too
    volatile uint64_t Bottom;
    volatile uint64_t Top;

    uint64_t ReadBottom() {
      //This read may need to be atomic
      return Bottom;
    }

    uint64_t ReadTop() {
      //This read may need to be atomic
      return Top;
    }

    void WriteBottom(uint64_t V) {
     //This write may need to be atomic
      Bottom = V;
    }
    void WriteTop(uint64_t V) {
     //This write may need to be atomic
      Top = V;
    }

    //Take advantage of the 48 bit virtual addresses on amd64

    void DecodeBottom(uint64_t v, DequeNode*& currNode, uint8_t& currIndex) {
      currNode  =      (DequeNode*)(v & 0x0000FFFFFFFFFFFFULL); 
      currIndex = (uint8_t)((v >> 48) & 0x0FFFF);
    }
    uint64_t EncodeBottom(DequeNode* currNode, uint8_t currIndex) {
      uint64_t v = 0;
      v  = (uint64_t)currNode;
      v |= (uint64_t)currIndex << 48;
      return v;
    }

    void DecodeTop(uint64_t v, uint8_t& currTopTag, DequeNode*& currTopNode, uint8_t& currTopIndex) {
      currTopNode  =      (DequeNode*)(v & 0x0000FFFFFFFFFFFFULL);
      currTopIndex = (uint8_t)((v >> 48) & 0x0FFFF);
      currTopTag   = (uint8_t)((v >> 56) & 0x0FFFF);
    }
    uint64_t EncodeTop(uint8_t currTopTag, DequeNode* currTopNode, uint8_t currTopIndex) {
      uint64_t v = 0;
      v  = (uint64_t)currTopNode;
      v |= (uint64_t)currTopIndex << 48;
      v |= (uint64_t)currTopTag << 56;
      return v;
    }

    bool CAS(volatile uint64_t* ptr, uint64_t old, uint64_t val) {
      return __sync_bool_compare_and_swap(ptr,old,val);
      /*
      unsigned char cas_result = 0;
      uint64_t old_low = (uint64_t)old;
      uint64_t old_high = (uint64_t)(old >> 64);
      uint64_t val_low = (uint64_t)val;
      uint64_t val_high = (uint64_t)(val >> 64);
 
      __asm__ __volatile__ 
	( 
	 // load *compare into into edx:eax 
	 // "mov eax, %;" 
	 // "mov edx, %;" 
	 // load *exchange into ecx:ebx 
	 // "mov ebx, %;" 
	 // "mov ecx, %;" 
	 "lock;"          // make cmpxchg16b atomic 
	 "cmpxchg16b %0;"  // cmpxchg16b sets ZF on success 
	 "setz      %1;"  // if ZF set, set cas_result to 1 
	 // output 
	 : "=m" (*ptr), "=q" (cas_result) 
	   // input 
	 : "m" (*ptr), "a" (old_low), "d" (old_high),
	   "b" (val_low), "c" (val_high) 
	   // clobbered 
	 : "cc", "memory" 
	  ); 
      //      std::cerr << std::hex << "CAS: " << (void*)ptr << " " << (uint64_t)((*ptr) >> 64) << " " << (uint64_t)*ptr << " " << old_high << " " << old_low << " " << val_high << " " << val_low << " {" << (int)cas_result << "}\n" << std::dec;
      return cas_result; 
      */
    }

    DequeNode* AllocateNode() {
      return new DequeNode;
    }

    bool emptinessTest(uint64_t bottomVal, uint64_t topVal) {
      DequeNode* botNode = 0;
      uint8_t botCellIndex = 0;
      DecodeBottom(bottomVal,botNode,botCellIndex);
      uint8_t topTag = 0;
      DequeNode* topNode = 0;
      uint8_t topCellIndex = 0;
      DecodeTop(topVal, topTag,topNode,topCellIndex);
      if ((botNode==topNode) && (botCellIndex==topCellIndex ||
				 botCellIndex==(topCellIndex+1))) {
	return true;
      } else if ((botNode==topNode->next) && (botCellIndex==0) &&
		 (topCellIndex==(DequeNode::ArraySize-1))) {
	return true;
      }
      return false;
    }

    //Local
    void PushBottom(T theData) {
      DequeNode* currNode = 0;
      uint8_t currIndex = 0;
      DequeNode* newNode = 0;
      uint8_t newIndex = 0;
      DecodeBottom(ReadBottom(),currNode, currIndex);
      currNode->itsDataArr[currIndex] = theData;
      //      std::cerr << "[" << currIndex << "] " << newIndex << "\n";
      //      std::cerr << "Wrote: " << theData << "\n";
      if (currIndex != 0) {
	newNode = currNode;
	newIndex = currIndex - 1;
      } else {
	newNode = AllocateNode();
	newNode->next = currNode;
	currNode->prev = newNode;
	newIndex = DequeNode::ArraySize - 1;
      }
      //	std::cerr << currIndex << " " << newIndex << "\n" << std::endl;
      WriteBottom(EncodeBottom(newNode,newIndex));
    }

    //Remote
    T PopTop(bool& EMPTY, bool& ABORT) {
      EMPTY = false;
      ABORT = false;
      uint64_t currTop = ReadTop();
      uint8_t currTopTag = 0;
      uint8_t currTopIndex = 0;
      DequeNode* currTopNode = 0;
      uint8_t newTopTag = 0;
      uint8_t newTopIndex = 0;
      DequeNode* newTopNode = 0;
      DequeNode* nodeToFree = 0;
      DecodeTop(currTop, currTopTag, currTopNode, currTopIndex);
      uint64_t currBottom = ReadBottom();
      if (emptinessTest(currBottom, currTop)) {
	if (currTop == ReadTop()) {
	  EMPTY = true;
	  return T();
	} else {
	  ABORT = true;
	  return T();
	}
      }
      if (currTopIndex != 0) {
	newTopTag = currTopTag;
	newTopNode = currTopNode;
	newTopIndex = currTopIndex - 1;
      } else {
	nodeToFree = currTopNode->next;
	newTopTag = currTopTag + 1;
	newTopNode = currTopNode->prev;
	newTopIndex = DequeNode::ArraySize - 1;
      }
      uint64_t newTopVal = EncodeTop(newTopTag, newTopNode, newTopIndex);
      T retVal = currTopNode->itsDataArr[currTopIndex];
      if (CAS(&Top, currTop, newTopVal)) {
	if (nodeToFree)
	  delete nodeToFree;
	return retVal;
      } else {
	ABORT = true;
	return T();
      }
    }

    //Local only
    bool Empty() {
      return emptinessTest(ReadBottom(), ReadTop());
    }

    //Local
    T PopBottom(bool& EMPTY) {
      EMPTY = false;
      DequeNode* oldBotNode = 0;
      uint8_t oldBotIndex = 0;
      DequeNode* newBotNode = 0;
      uint8_t newBotIndex = 0;
      uint64_t oldBotVal = ReadBottom();
      DecodeBottom(oldBotVal, oldBotNode, oldBotIndex);
      if (oldBotIndex != DequeNode::ArraySize-1) {
	newBotNode = oldBotNode;
	newBotIndex = oldBotIndex+1;
      } else {
	newBotNode = oldBotNode->next;
	newBotIndex = 0;
      }
      //      std::cerr << oldBotIndex << " [" << newBotIndex << "]\n";
      uint64_t newBotVal = EncodeBottom(newBotNode,newBotIndex);
      WriteBottom(newBotVal);
      uint64_t currTop = ReadTop();
      uint8_t currTopTag = 0;
      DequeNode* currTopNode = 0;
      uint8_t currTopIndex = 0;
      DecodeTop(currTop, currTopTag,currTopNode,currTopIndex);
      T retVal = newBotNode->itsDataArr[newBotIndex]; // Read data to be popped
      //      std::cerr << "Read: " << retVal << "\n";
      if (oldBotNode == currTopNode && oldBotIndex == currTopIndex ) { 
	// Case 1: if Top has crossed Bottom
	//Return bottom to its old possition:
	WriteBottom(EncodeBottom(oldBotNode,oldBotIndex));
	EMPTY = true;
	//	std::cerr << "Returning empty 1\n";
	return T();
      } else if ( newBotNode == currTopNode && newBotIndex == currTopIndex ) {
	// Case 2: When popping the last entry
	//in the deque (i.e. deque is
	//empty after the update of bottom).
	//Try to update Top’s tag so no concurrent PopTop operation will also pop the same entry:
	uint64_t newTopVal = EncodeTop(currTopTag+1, currTopNode, currTopIndex);
	if (CAS(&Top, currTop, newTopVal)) {
	  if (oldBotNode != newBotNode)
	    delete oldBotNode;
	  return retVal;
	} else {
	  // if CAS failed (i.e. a concurrent PopTop operation already popped the last entry):
	  //Return bottom to its old possition:
	  WriteBottom(EncodeBottom(oldBotNode,oldBotIndex));
	  EMPTY = true;
	  //	  std::cerr << "Returning empty 2\n";
	  return T();
	}
      } else {
	// Case 3: Regular case (i.e. there was at least one entry in the deque after bottom’s update):
	if (oldBotNode != newBotNode)
	  delete oldBotNode;
	return retVal;
      }
    }

  public:

    GWL_ChaseLev_Dyn()
      :Bottom(0), Top(0)
    {
      DequeNode* nodeA = AllocateNode();
      DequeNode* nodeB = AllocateNode();
      nodeA->prev = 0;
      nodeA->next = nodeB;
      nodeB->next = 0;
      nodeB->prev = nodeA;
      int newIndex = DequeNode::ArraySize - 1;
      WriteBottom(EncodeBottom(nodeA,newIndex));
      WriteTop(EncodeTop(0,nodeA,newIndex));
    }
      

    //These should only be called by one thread
    virtual void push(T val) {
      PushBottom(val);
    }

    T pop(bool& succeeded) {
      bool Emp;
      T retval = PopBottom(Emp);
      succeeded = !Emp;
      return retval;
    }
    
    //This can be called by any thread
    T steal(bool& succeeded) {
      bool Empty, Abort;
      T retval = PopTop(Empty,Abort);
      succeeded = !(Empty || Abort);
      return retval;
    }

    bool empty() {
      return Empty();
    }

  };

  template<typename T>
  class GWL_Idempotent_LIFO : public Galois::WorkList<T> {

    NOCOPY(GWL_Idempotent_LIFO);

    packedInt2<32,32> anchor; //tail,tag
    unsigned int capacity;
    T* volatile tasks;
    
    inline void order() {
      //Compiler barier
      __asm__("":::"memory");
    }

    bool Empty() {
      unsigned int t,g;
      anchor.packedRead(t,g);
      return t == 0;
    }

    void put(T task) {
      //Order write in 3 before write in 4
      unsigned int t,g;
      anchor.packedRead(t,g);
      if (t == capacity) {
	expand();
	put(task);
	return;
      }
      tasks[t] = task;
      order();
      anchor.packedWrite(t+1,g+1);
    }
    
    T take(bool& EMPTY) {
      EMPTY = false;
      unsigned int t,g;
      anchor.packedRead(t,g);
      if (t == 0) {
	EMPTY = true;
	return T();
      }
      T task = tasks[t-1];
      anchor.packedWrite(t-1,g);
      return task;
    }
    
    T i_steal(bool& EMPTY) {
      EMPTY = false;
      //Order read in 1 before read in 3
      //Order read in 4 before CAS in 5
      unsigned int t,g;
      anchor.packedRead(t,g);
      if (t == 0) {
	EMPTY = true;
	return T();
      }
      order();
      T* a = tasks;
      T task = a[t-1];
      order();
      if (!anchor.CAS(t,g, t-1,g )) {
	return i_steal(EMPTY);
      }
      return task;
    }
    
    void expand() {
      //Order writes in 2 before write in 3
      //Order write in 3 before write in put:4
      T* a = new T[2*capacity];
      for( int i = 0; i < (int)capacity; ++i)
	a[i] = tasks[i];
      order();
      tasks = a;
      capacity = 2*capacity;
      order();
    }
   
  public:
    GWL_Idempotent_LIFO(int size = 256)
      :anchor(0,0), capacity(size)
    {
      tasks = new T[size];
    }

    virtual void push(T val) {
      put(val);
    }

    T pop(bool& succeeded) {
      bool Empty;
      T retval = take(Empty);
      succeeded = !Empty;
      return retval;
    }
    
    //This can be called by any thread
    T steal(bool& succeeded) {
      bool Empty;
      T retval = i_steal(Empty);
      succeeded = !Empty;
      return retval;
    }

    bool empty() {
      return Empty();
    }

  };


  template<typename T>
  class GWL_Idempotent_FIFO : public Galois::WorkList<T> {

    NOCOPY(GWL_Idempotent_FIFO);

    struct TaskArrayWithSize {
      int size;
      T array[1];
    };

    TaskArrayWithSize* mkArray(int num) {
      TaskArrayWithSize* r = (TaskArrayWithSize*)malloc(sizeof(TaskArrayWithSize)+sizeof(T[num]));
      r->size = num;
      return r;
    }

    int head;
    int tail;
    TaskArrayWithSize* volatile tasks;
    
    inline void order() {
      //Compiler barier
      __asm__("":::"memory");
    }

    bool Empty() {
      return head == tail;
    }

    void put(T task) {
      //Order write at 4 before write at 5
      int h = head;
      int t = tail;
      if (t == h+tasks->size) {
	expand();
	put(task);
	return;
      }
      tasks->array[t%tasks->size] = task;
      order();
      tail = t+1;
    }
    
    T take(bool& EMPTY) {
      EMPTY = false;
      int h = head;
      int t = tail;
      if (h == t) {
	EMPTY = true;
	return T();
      }
      T task = tasks->array[h%tasks->size];
      head = h+1;
      return task;
    }
    
    T i_steal(bool& EMPTY) {
      EMPTY = false;
      //Order read in 1 before read in 2
      //Order read in 1 before read in 4
      //Order read in 5 before CAS in 6
      int h = head;
      order();
      int t = tail;
      order();
      if (h == t) {
	EMPTY = true;
	return T();
      }
      TaskArrayWithSize* a = tasks;
      T task = a->array[h%a->size];
      order();
      if (!__sync_bool_compare_and_swap(&head,h,h+1)) {
	return i_steal(EMPTY);
      }
      return task;
    }
    
    void expand() {
      //Order writes in 2 and 4 before write in 5
      //Order write in 5 before write in put:5
      int size = tasks->size;
      TaskArrayWithSize* a = mkArray(2*size);
      for (int i = head; i < tail; ++i)
	a->array[i%a->size] = tasks->array[i%tasks->size];
      order();
      tasks = a;
      order();
    }
   
  public:
    GWL_Idempotent_FIFO(int size = 256)
      :head(0), tail(0)
    {
      tasks = mkArray(size);
    }

    virtual void push(T val) {
      put(val);
    }

    T pop(bool& succeeded) {
      bool Empty;
      T retval = take(Empty);
      succeeded = !Empty;
      return retval;
    }
    
    //This can be called by any thread
    T steal(bool& succeeded) {
      bool Empty;
      T retval = i_steal(Empty);
      succeeded = !Empty;
      return retval;
    }

    bool empty() {
      return Empty();
    }

  };


  template<typename T>
  class GWL_Idempotent_FIFO_SB : public Galois::WorkList<T> {

    NOCOPY(GWL_Idempotent_FIFO_SB);

    struct TaskArrayWithSize {
      int size;
      T array[1];
    };

    TaskArrayWithSize* mkArray(int num) {
      TaskArrayWithSize* r = (TaskArrayWithSize*)malloc(sizeof(TaskArrayWithSize)+sizeof(T[num]));
      r->size = num;
      return r;
    }

    packedInt3<21,21,22> anchor;
    TaskArrayWithSize* volatile tasks;
    
    inline void order() {
      //Compiler barier
      __asm__("":::"memory");
    }


    bool Empty() {
      unsigned int h,s,g;
      anchor.packedRead(h,s,g);
      return s == 0;
    }

    void put(T task) {
      //Order write in 3 before write in 4
      unsigned int h,s,g;
      anchor.packedRead(h,s,g);
      if ((int)s == tasks->size) {
	expand();
	put(task);
	return;
      }
      tasks->array[(h+s)%tasks->size] = task;
      order();
      anchor.packedWrite(h,s+1,g+1);
    }
    
    T take(bool& EMPTY) {
      EMPTY = false;
      unsigned int h,s,g;
      anchor.packedRead(h,s,g);
      if (s == 0) {
	EMPTY = true;
	return T();
      }
      T task = tasks->array[(h+s-1)%tasks->size];
      anchor.packedWrite(h,s-1,g);
      return task;
    }
    
    T i_steal(bool& EMPTY) {
      EMPTY = false;
      //Order read in 1 before read in 3
      //Order read in 4 before CAS in 6
      unsigned int h,s,g;
      anchor.packedRead(h,s,g);
      if (s == 0) {
	EMPTY = 0;
	return T();
      }
      order();
      TaskArrayWithSize* a = tasks;
      T task = a->array[h%a->size];
      unsigned int h2 = (h+1) % a->size;
      order();
      if (!anchor.CAS(h,s,g , h2,s-1,g )) {
	return i_steal(EMPTY);
      }
      return task;
    }
    
    void expand() {
      //Order writes in 2 and 4 before write in 5
      //Order write in 5 before write in put:4
      unsigned int h,s,g;
      anchor.packedRead(h,s,g);
      TaskArrayWithSize* a = mkArray(2*s);
      for (unsigned int i = 0; i < s; ++i)
	a->array[(h+i)%a->size] = tasks->array[(h+i)%tasks->size];
      order();
      tasks = a;
      order();
    }
   
  public:
    GWL_Idempotent_FIFO_SB(int size = 256)
      :anchor(0,0,0)
    {
      tasks = mkArray(size);
    }

    virtual void push(T val) {
      put(val);
    }

    T pop(bool& succeeded) {
      bool Empty;
      T retval = take(Empty);
      succeeded = !Empty;
      return retval;
    }
    
    //This can be called by any thread
    T steal(bool& succeeded) {
      bool Empty;
      T retval = i_steal(Empty);
      succeeded = !Empty;
      return retval;
    }

    bool empty() {
      return Empty();
    }

  };

}
