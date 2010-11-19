/* -*- C++ -*- */

/*
  The Hoard Multiprocessor Memory Allocator
  www.hoard.org

  Author: Emery Berger, http://www.cs.umass.edu/~emery
 
  Copyright (c) 1998-2006 Emery Berger, The University of Texas at Austin

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

// Link this code with your executable to use winhoard.

#include <windows.h>
#if defined(_WIN64)
#pragma comment(linker, "/include:ReferenceHoard")
#else
#pragma comment(linker, "/include:_ReferenceHoard")
#endif

#include <stdio.h>

extern "C" {

  __declspec(dllimport) int ReferenceHoardStub;
  
  void ReferenceHoard (void)
  {
    LoadLibraryA ("winhoard.dll");
    ReferenceHoardStub = 1; 
  }

}
