/* -*- C++ -*- */

/*
  The Hoard Multiprocessor Memory Allocator
  www.hoard.org

  Author: Emery Berger, http://www.cs.umass.edu/~emery
 
  Copyright (c) 1998-2008 Emery Berger, The University of Texas at Austin

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

#ifdef _DEBUG
#error "This library must be compiled in release mode."
#endif	

/*

  Compile with compile-winhoard.cmd.

 */


#include <windows.h>

#define WIN32_LEAN_AND_MEAN

#if (_WIN32_WINNT < 0x0500)
#define _WIN32_WINNT 0x0500
#endif

#pragma inline_depth(255)

#pragma warning(disable: 4273)
#pragma warning(disable: 4098)  // Library conflict.
#pragma warning(disable: 4355)  // 'this' used in base member initializer list.
#pragma warning(disable: 4074)	// initializers put in compiler reserved area.

void (*hoard_memcpy_ptr)(void *dest, const void *src, size_t count);
void (*hoard_memset_ptr)(void *dest, int c, size_t count);

const char *RlsCRTLibraryName[] = {"MSVCR71.DLL", "MSVCR80.DLL", "MSVCR90.DLL", "MSVCR100.DLL" };
static const int RlsCRTLibraryNameLength = 4;

static const int BytesToStore = 5;

#define IAX86_NEARJMP_OPCODE	  0xe9
#define MakeIAX86Offset(to,from)  ((unsigned)((char*)(to)-(char*)(from)) - BytesToStore)

typedef struct
{
  const char *import;		// import name of patch routine
  FARPROC replacement;		// pointer to replacement function
  FARPROC original;		// pointer to original function
  unsigned char codebytes[BytesToStore];	// original code storage
} PATCH;


static bool PatchMeIn (void);

#define CUSTOM_PREFIX(n) hoard_##n
#define HOARD_PRE_ACTION {PatchMeIn();}
#define HOARD_POST_ACTION {HeapAlloc (GetProcessHeap(), 0, 1); }

// DisableThreadLibraryCalls ((HMODULE)hinstDLL); 

#define CUSTOM_DLLNAME HoardDllMain

#include "libhoard.cpp"

// Intercept the exit functions.

static const int HOARD_MAX_EXIT_FUNCTIONS = 255;
static int exitCount = 0;

extern "C" {

  __declspec(dllexport) int ReferenceHoardStub;

  typedef void (*exitFunctionType) (void);
  exitFunctionType exitFunctionBuffer[255];

  void hoard_onexit (void (*function)(void)) {
    if (exitCount < HOARD_MAX_EXIT_FUNCTIONS) {
      exitFunctionBuffer[exitCount] = function;
      exitCount++;
    }
  }

  void hoard_exit (int code) {
    while (exitCount > 0) {
      exitCount--;
      (exitFunctionBuffer[exitCount])();
    }
  }

  void * hoard_expand (void * ptr) {
    return NULL;
  }

}


/* ------------------------------------------------------------------------ */

static PATCH rls_patches[] = 
  {
    // RELEASE CRT library routines supported by this memory manager.

    {"_expand",		(FARPROC) hoard_expand,	   0},
    {"_onexit",         (FARPROC) hoard_onexit,    0},
    {"_exit",           (FARPROC) hoard_exit,      0},

    // operator new, new[], delete, delete[].

#ifdef _WIN64

    {"??2@YAPEAX_K@Z",  (FARPROC) hoard_malloc,    0},
    {"??_U@YAPEAX_K@Z", (FARPROC) hoard_malloc,    0},
    {"??3@YAXPEAX@Z",   (FARPROC) hoard_free,      0},
    {"??_V@YAXPEAX@Z",  (FARPROC) hoard_free,      0},

#else

    {"??2@YAPAXI@Z",    (FARPROC) hoard_malloc,    0},
    {"??_U@YAPAXI@Z",   (FARPROC) hoard_malloc,    0},
    {"??3@YAXPAX@Z",    (FARPROC) hoard_free,      0},
    {"??_V@YAXPAX@Z",   (FARPROC) hoard_free,      0},

#endif

    // the nothrow variants new, new[].

    {"??2@YAPAXIABUnothrow_t@std@@@Z",  (FARPROC) hoard_malloc, 0},
    {"??_U@YAPAXIABUnothrow_t@std@@@Z", (FARPROC) hoard_malloc, 0},

    {"_msize",	(FARPROC) hoard_malloc_usable_size,		0},
    {"calloc",	(FARPROC) hoard_calloc,		0},
    {"malloc",	(FARPROC) hoard_malloc,		0},
    {"realloc",	(FARPROC) hoard_realloc,		0},
    {"free",	(FARPROC) hoard_free,              0},
    {"_recalloc", (FARPROC) hoard_recalloc, 0},
    {"_putenv",	(FARPROC) hoard__putenv,		0},
    {"getenv",	(FARPROC) hoard_getenv,		0},
  };


static void PatchIt (PATCH *patch)
{
  // Change rights on CRT Library module to execute/read/write.

  MEMORY_BASIC_INFORMATION mbi_thunk;
  VirtualQuery((void*)patch->original, &mbi_thunk, 
	       sizeof(MEMORY_BASIC_INFORMATION));
  VirtualProtect(mbi_thunk.BaseAddress, mbi_thunk.RegionSize, 
		 PAGE_EXECUTE_READWRITE, &mbi_thunk.Protect);

  // Patch CRT library original routine:
  // 	save original code bytes for exit restoration
  //		write jmp <patch_routine> (at least 5 bytes long) to original.

  (*hoard_memcpy_ptr)(patch->codebytes, patch->original, sizeof(patch->codebytes));
  unsigned char *patchloc = (unsigned char*)patch->original;
  *patchloc++ = IAX86_NEARJMP_OPCODE;
  *(unsigned*)patchloc = MakeIAX86Offset(patch->replacement, patch->original);
	
  // Reset CRT library code to original page protection.

  VirtualProtect(mbi_thunk.BaseAddress, mbi_thunk.RegionSize, 
		 mbi_thunk.Protect, &mbi_thunk.Protect);
}


static bool PatchMeIn (void)
{
  bool patchedIn = false;
  
  // acquire the module handles for the CRT libraries (release and debug)
  for (int i = 0; i < RlsCRTLibraryNameLength; i++) {

    HMODULE RlsCRTLibrary = GetModuleHandle(RlsCRTLibraryName[i]);

    HMODULE DefCRTLibrary = 
      RlsCRTLibrary;

    // assign function pointers for required CRT support functions
    if (DefCRTLibrary) {
      hoard_memcpy_ptr = (void(*)(void*,const void*,size_t))
	GetProcAddress(DefCRTLibrary, "memcpy");
      hoard_memset_ptr = (void(*)(void*,int,size_t))
	GetProcAddress(DefCRTLibrary, "memset");
    }

    // patch all relevant Release CRT Library entry points
    if (RlsCRTLibrary) {
      for (int i = 0; i < sizeof(rls_patches) / sizeof(*rls_patches); i++) {
	if (rls_patches[i].original = GetProcAddress(RlsCRTLibrary, rls_patches[i].import)) {
	  PatchIt(&rls_patches[i]);
 	  patchedIn = true;
	}
      }
    }
  }
  return patchedIn;
}

// And now, again.

#define CUSTOM_PREFIX(n) n
#include "wrapper.cpp"
