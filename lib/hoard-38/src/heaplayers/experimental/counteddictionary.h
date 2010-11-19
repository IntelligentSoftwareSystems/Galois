#ifndef _COUNTEDDICTIONARY_H_
#define _COUNTEDDICTIONARY_H_

template <class Dict>
class CountedDictionary : public Dict {
public:
  class Entry : public Dict::Entry {};

  __forceinline CountedDictionary (void)
    : num (0)
  {}

  __forceinline void clear (void) {
    num = 0;
    Dict::clear();
  }

  __forceinline Entry * get (void) {
    Entry * e = (Entry *) Dict::get();
    if (e) {
      --num;
    }
    return e;
  }

  __forceinline Entry * remove (void) {
    Entry * e = (Entry *) Dict::remove();
    if (e) {
      --num;
    }
    return e;
  }

  __forceinline void insert (Entry * e) {
    Dict::insert (e);
    ++num;
  }

  __forceinline int getNumber (void) const {
    return num;
  }

private:
  int num;
};

#endif
