#ifndef _PAD_H_
#define _PAD_H_


// Add a cache-pad to an object.

template <int CacheLineSize, class Super>
class Pad : public Super {
private:
	// Add a pad field with a name that is unlikely
	// to conflict with a real field name.
	char _pad_QWERTYUIOP1234567890[CacheLineSize];
};


#endif
