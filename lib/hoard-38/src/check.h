// -*- C++ -*-

#ifndef _CHECK_H_
#define _CHECK_H_

template <class TYPE, class CHECK>
class Check {
public:
  Check (TYPE * t)
#ifndef NDEBUG
    : _object (t)
#endif
  {
    t = t; // avoid warning
#ifndef NDEBUG
    CHECK::precondition (_object);
#endif
  }

  ~Check (void) {
#ifndef NDEBUG
    CHECK::postcondition (_object);
#endif
  }

private:
  Check (const Check&);
  Check& operator=(const Check&);

#ifndef NDEBUG
  TYPE * _object;
#endif

};

#endif
