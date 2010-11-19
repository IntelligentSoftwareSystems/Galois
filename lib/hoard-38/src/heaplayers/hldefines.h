#ifndef _HLDEFINES_H_
#define _HLDEFINES_H_

/*
 * @file hldefines.h
 * @brief Defines macros used throughout Heap Layers.
 *
 */

// Define HL_EXECUTABLE_HEAP as 1 if you want that (i.e., you're doing
// dynamic code generation).

#define HL_EXECUTABLE_HEAP 0

#if defined(_MSC_VER)

// Microsoft Visual Studio
#pragma inline_depth(255)
#define INLINE __forceinline
#define inline __forceinline
#define NO_INLINE __declspec(noinline)
#pragma warning(disable: 4530)
#define MALLOC_FUNCTION
#define RESTRICT

#elif defined(__GNUC__)

// GNU C

#define NO_INLINE       __attribute__ ((noinline))
#define INLINE          inline
#define MALLOC_FUNCTION __attribute__((malloc))
#define RESTRICT        __restrict__

#else

// All others

#define NO_INLINE
#define INLINE inline
#define MALLOC_FUNCTION
#define RESTRICT

#endif

#endif
