from cython.operator cimport preincrement, dereference as deref
from libgalois.Galois cimport UserContext, iterate, for_each, setActiveThreads, SharedMemSys, loopname, no_conflicts, no_pushes, gPrint, do_all, GReduceMax, InsertBag, steal
from libgalois.Galois cimport LargeArray, MethodFlag, FLAG_UNPROTECTED
from libgalois.graphs.Graph cimport dummy_true, dummy_false, MorphGraph, LC_CSR_Graph
from libgalois.graphs.Util cimport readGraph
from libgalois.Timer cimport Timer
from libstd.atomic cimport atomic
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
import sys
from libc.stdint cimport *
from libc.math cimport fabs

# Initialize the Galois runtime when the Python module is loaded.
cdef class _galois_runtime_wrapper:
    cdef SharedMemSys _galois_runtime

cdef extern from * nogil:
    # hack to bind leading arguments by value to something that can be passed
    # to for_each. The returned lambda needs to be usable after the scope
    # where it is created closes, so captured values are captured by value.
    # The by-value capture in turn requires that graphs be passed as
    # pointers. This function is used without exception specification under
    # the assumption that it will always be used as a subexpression of
    # a whole expression that requires exception handling or that it will
    # be used in a context where C++ exceptions are appropriate.
    # There are more robust ways to do this, but this didn't require
    # users to find and include additional C++ headers specific to
    # this interface.
    # Syntactically, this is using the cname of an "external" function
    # to create a one-line macro that can be used like a function.
    # The expected use is bind_leading(function, args).
    cdef void *bind_leading "[](auto f, auto&&... bound_args){return [=](auto&&... pars){return f(bound_args..., pars...);};}"(...)
    # Similar thing to invoke a function and return an integer.
    # Useful for verifying that this approach works.
    cdef int invoke "[](auto f, auto&&... args){return f(args...);}"(...)

#cdef int myfunc(int a, int b, int c):
#    return a + b + c

cdef extern from "algorithm" namespace "std" nogil:
    # This function from <algorithm> isn't currently
    # provided by Cython's known interfaces for the C++ standard library,
    # so this is needed to get it working here.
    # The variadic signature could probably be removed and this could
    # be made to match the original templates more closely, but since
    # this form matches the syntax we need to use, it is good enough.
    int count_if(...) except +

# This function is expected to forward C++ exceptions thrown to
# its caller. This is unusual for Cython, but it's the simplest
# way to guarantee no loos Python exceptions end up floating around.
#cdef void IncrementNeighbors(Graph *g, GNode n, UserContext[GNode] &ctx) nogil:
 #   cdef:
#        MorphGraph[int, void, dummy_true].edge_iterator ii = g[0].edge_begin(n)
#        MorphGraph[int, void, dummy_true].edge_iterator ei = g[0].edge_end(n)
#        int *data
#    while ii != ei:
#        data = &g[0].getData(g[0].getEdgeDst(ii))
#        preincrement(data[0])
#        preincrement(ii)

# C++ exceptions thrown inside this function are forwarded to its caller.
#cdef bint ValueEqual(Graph *g, int v, GNode n) nogil:
#    return g[0].getData(n) == v

#cdef bint SameNodes(GNodeCSR n, GNodeCSR s) nogil:
 #   return n == s

#cdef void setGNode(Graph_CSR *g, GNodeCSR n, int val) nogil:
#    gPrint("inside setGNode\n")
#    cdef uint32_t *data = &g[0].getData(n);
#    data[0] = val;
#    gPrint("n : ", deref(data), "\n");
#    preincrement(data[0])
#    gPrint("n : ", deref(data), "\n");



