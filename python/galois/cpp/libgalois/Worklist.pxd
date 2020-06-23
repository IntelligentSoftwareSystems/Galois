from libc.stdint cimport *

cdef extern from *:
    cppclass dummy_true "true"
    cppclass dummy_false "false"
    cppclass Uint_64u "64u"

##
# TODO: Need a better way to provide user defined
# functions as template parameters to DS such as
# OBIM
##
cdef extern from "galois/Constants.h" namespace "galois" nogil:
    cppclass UpdateRequestIndexer:
        UpdateRequestIndexer(uint32_t)
        pass
    cppclass UpdateRequest[G, D]:
        G src
        D dist
        UpdateRequest(G&, D)
        pass
    cppclass ReqPushWrap:
        pass
        
cdef extern from "galois/Traits.h" namespace "galois" nogil:
    cppclass s_wl:
        pass
    s_wl wl[T](...)

cdef extern from "galois/worklists/Chunk.h" namespace "galois::worklists" nogil:
    cppclass ChunkFIFO[T]:
        pass
    cppclass PerSocketChunkFIFO[T]:
        pass

cdef extern from "galois/worklists/Obim.h" namespace "galois::worklists" nogil:
    cppclass OrderedByIntegerMetric[UpdateFuncTy, WorkListTy]:
        pass


        