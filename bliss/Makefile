CFLAGS = -I.
CFLAGS += -g
#CFLAGS += -pg
CFLAGS += -Wall
CFLAGS += --pedantic
CFLAGS += -O9
#CFLAGS += -DBLISS_DEBUG
CFLAGS += -fPIC

SRCS = defs.cc graph.cc partition.cc orbit.cc uintseqhash.cc heap.cc
SRCS += timer.cc utils.cc bliss_C.cc

OBJS = $(addsuffix .o, $(basename $(SRCS)))

GMPOBJS = $(addsuffix g,  $(OBJS))

LIB =
#LIB += /usr/lib/ccmalloc.o -ldl

CC = g++
RANLIB = ranlib
AR = ar
BLISSLIB = libbliss.a

gmp:	LIB += -lgmp
gmp:	CFLAGS += -DBLISS_USE_GMP

normal:	bliss
gmp:	bliss_gmp


all:: lib bliss

%.o %.og:	%.cc
	$(CC) $(CFLAGS) -c -o $@ $<

lib: $(OBJS)
	rm -f $(BLISSLIB)
	$(AR) cr $(BLISSLIB) $(OBJS)
	$(RANLIB) $(BLISSLIB)

lib_gmp: $(GMPOBJS)
	rm -f $(BLISSLIB)
	$(AR) cr $(BLISSLIB) $(GMPOBJS)
	$(RANLIB) $(BLISSLIB)

bliss: bliss.o lib $(OBJS)
	$(CC) $(CFLAGS) -o bliss bliss.o $(OBJS) $(LIB)

bliss_gmp: bliss.og lib_gmp $(GMPOBJS)
	$(CC) $(CFLAGS) -o bliss bliss.og $(GMPOBJS) $(LIB)


clean:
	rm -f bliss $(BLISSLIB) $(OBJS) bliss.o $(GMPOBJS) bliss.og

# DO NOT DELETE
