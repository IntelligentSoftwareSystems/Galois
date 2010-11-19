// Report stats.

#include <stdio.h>

#include "timer.h"

struct mallinfo {
  int arena;    /* non-mmapped space allocated from system */
  int ordblks;  /* number of free chunks */
  int smblks;   /* number of fastbin blocks */
  int hblks;    /* number of mmapped regions */
  int hblkhd;   /* space in mmapped regions */
  int usmblks;  /* maximum total allocated space */
  int fsmblks;  /* space available in freed fastbin blocks */
  int uordblks; /* total allocated space */
  int fordblks; /* total free space */
  int keepcost; /* top-most, releasable (via malloc_trim) space */
};

#if 1
extern "C" struct mallinfo dlmallinfo (void);
#endif
extern "C" void mstats (void); // for Kingsley malloc.
extern "C" void dlmalloc_stats (void); // for Doug Lea malloc.
extern "C" void * sbrk (long);
extern "C" void malloc_stats (void); // for Doug Lea malloc.

int masMem;// FIX ME


int totalAllocated;
int maxAllocated;
int totalRequested;
int maxRequested;

class TimeItAll {
public:
	TimeItAll (void) {
		totalAllocated = 0;
		maxAllocated = 0;
		t.start();
		// FIX ME
		masMem = 0;
#ifdef WIN32
		initSpace = (long) sbrk(0);
		//initSpace = CommittedSpace();
#endif
	}

	~TimeItAll (void) {
		t.stop();
#if 0
		struct mallinfo mi = dlmallinfo();
		printf ("Doug Lea Memory allocated = {total} %d, {non-mmapped} %d\n", mi.usmblks, mi.arena);
#endif
		double elapsed = (double) t;
		printf ("Time elapsed = %f\n", elapsed);
		printf ("Max allocated = %d\n", maxAllocated);
		printf ("Max requested = %d\n", maxRequested);
		printf ("Frag = %lf\n", (double) maxAllocated / (double) maxRequested);
#ifdef WIN32
		printf ("init space = %ld\n",  initSpace);
		printf ("Sbrk report: %ld\n", (long) sbrk(0));
#if 0
		size_t diff = (size_t) wsbrk(8);
		cout << "diff1 = " << diff << endl;
		diff -= initSpace - 8;
		cout << "diff = " << diff << endl;
#endif
#endif
		malloc_stats();
		fflush (stdout);
//		dlmalloc_stats();
	}
private:
	Timer t;
#ifdef WIN32
	long initSpace;
#endif
};

TimeItAll timer;
