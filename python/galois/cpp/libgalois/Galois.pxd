
from libcpp cimport bool
from libc.stdint cimport *

# Declaration from "Galois/Threads.h"

#ctypedef uint64_t size_t 

# Hack to make auto return type for galois::iterate work.
# It may be necessary to write a wrapper header around for_each,
# but this should be good enough for the forseeable future either way.
cdef extern from * nogil:
    cppclass InternalRange "auto":
        pass

cdef extern from "galois/Galois.h" namespace "galois" nogil:
    unsigned int setActiveThreads(unsigned int)
    void gPrint(...)
    cppclass UserContext[T]:
        pass

    void for_each(...)
    void do_all(...)

    InternalRange iterate[T](T &, T &)
    InternalRange iterate[T](T &)

    cppclass SharedMemSys:
        SharedMemSys()

    cppclass loopname:
        loopname(char *name)

    cppclass no_pushes:
        no_pushes()

    cppclass steal:
        steal()

    cppclass no_conflicts:
        no_conflicts()

    cppclass GReduceMax[T]:
        pass
        void update(T)
        T reduce()
        void reset()

    cppclass InsertBag[T]:
        pass
        void push(T)
        bool empty()
        void swap(InsertBag&)
        void clear()

    cppclass LargeArray[T]:
        pass
        void allocateInterleaved(size_t)
        void allocateBlocked(size_t)
        T &operator[](size_t)

cdef extern from "galois/MethodFlags.h" namespace "galois" nogil:
    cdef cppclass MethodFlag:
        pass
    
    cdef MethodFlag FLAG_UNPROTECTED "galois::MethodFlag::UNPROTECTED"
    cdef MethodFlag FLAG_WRITE "galois::MethodFlag::WRITE"
    cdef MethodFlag FLAG_READ "galois::MethodFlag::READ"
    cdef MethodFlag FLAG_INTERNAL_MASK "galois::MethodFlag::INTERNAL_MASK"
    cdef MethodFlag PREVIOUS "galois::MethodFlag::PREVIOUS"

    
