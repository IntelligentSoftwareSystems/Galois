// -*- C++ -*-

#ifndef _ALIGNMENTOF_H_
#define _ALIGNMENTOF_H_

template <class T>
class alignmentOf {
private:
  struct alignmentFinder {
    char a;
    T b;
  };
public:
  enum { value = sizeof(alignmentFinder) - sizeof(T) };
};

#endif // _ALIGNMENTOF_H_

