cdef extern from "galois/Timer.h" namespace "galois" nogil:
    cppclass Timer:
        Timer()
        void start()
        void stop()
        unsigned int get()
