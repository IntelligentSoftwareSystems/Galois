/**
 * @file tls.cpp
 * @author Emery Berger <http://www.cs.umass.edu/~emery>
 *
 * This file leverages compiler support for thread-local variables for
 * access to thread-local heaps, when available. It also intercepts
 * thread completions to flush these local heaps, returning any unused
 * memory to the global Hoard heap. On Windows, this happens in
 * DllMain. On Unix platforms, we interpose our own versions of
 * pthread_create and pthread_exit.
*/


#if defined(_WIN32)
#include "wintls.cpp"
#else
#include "unixtls.cpp"
#endif

