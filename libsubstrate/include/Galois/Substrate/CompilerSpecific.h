/** Galois configuration -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Factor out compiler-specific lowlevel stuff
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_SUBSTRATE_COMPILERSPECIFIC_H
#define GALOIS_SUBSTRATE_COMPILERSPECIFIC_H

namespace Galois {
namespace Substrate {

inline static void asmPause() {
#if defined(__i386__) || defined(__amd64__)
  //  __builtin_ia32_pause();
  asm volatile ("pause"); 
#endif
}

inline static void compilerBarrier() {
  asm volatile ("":::"memory");
}

inline static void flushInstructionPipeline() {
#if defined(__i386__) || defined(__amd64__)
  asm volatile (
      "xor %%eax, %%eax;"
      "cpuid;"
      :::"%eax", "%ebx", "%ecx", "%edx");
#endif
}

//xeons have 64 byte cache lines, but will prefetch 2 at a time
#define GALOIS_CACHE_LINE_SIZE 128

#if defined(__INTEL_COMPILER)
#define GALOIS_ATTRIBUTE_NOINLINE __attribute__ ((noinline))
#define GALOIS_ATTRIBUTE_DEPRECATED __attribute__ ((deprecated))
#define GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE __attribute__((aligned(GALOIS_CACHE_LINE_SIZE)))

#elif defined( __GNUC__)
#define GALOIS_ATTRIBUTE_NOINLINE __attribute__ ((noinline))
#define GALOIS_ATTRIBUTE_DEPRECATED __attribute__ ((deprecated))
#define GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE __attribute__((aligned(GALOIS_CACHE_LINE_SIZE)))

#elif defined( _MSC_VER)
#define GALOIS_ATTRIBUTE_NOINLINE __declspec(noinline)
#define GALOIS_ATTRIBUTE_DEPRECATED __declspec ((deprecated))
#define GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE __declspec(align(GALOIS_CACHE_LINE_SIZE))

#else
#define GALOIS_ATTRIBUTE_NOINLINE
#define GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE
#define GALOIS_ATTRIBUTE_DEPRECATED __attribute__ ((deprecated))
#endif

// used to disable inlining of functions so that they
// show up in stack samples when profiling
#ifdef GALOIS_USE_PROF
#define GALOIS_ATTRIBUTE_PROF_NOINLINE GALOIS_ATTRIBUTE_NOINLINE
#else
#define GALOIS_ATTRIBUTE_PROF_NOINLINE inline
#endif

} // end namespace Substrate
} // end namespace Galois

#endif
