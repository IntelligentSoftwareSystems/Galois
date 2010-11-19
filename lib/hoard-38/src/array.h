// -*- C++ -*-

#ifndef _ARRAY_H_
#define _ARRAY_H_

/*

  Heap Layers: An Extensible Memory Allocation Infrastructure
  
  Copyright (C) 2000-2003 by Emery Berger
  http://www.cs.umass.edu/~emery
  emery@cs.umass.edu
  
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

#include <cassert>

namespace Hoard {

template <int N, typename T>
class Array {
public:

  inline T& operator()(int index) {
    assert (index >= 0);
    assert (index < N);
    return _item[index];
  }

  inline const T& operator()(int index) const {
    assert (index >= 0);
    assert (index < N);
    return _item[index];
  }

private:

  T _item[N];

};

}


#endif
