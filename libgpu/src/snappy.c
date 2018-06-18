#include <snappy-c.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "snfile.h"
#include <errno.h>
#include <string.h>

const int BUFSIZE = 10485760;

struct snappy_file {
  FILE* f;
  char* buf;
  size_t bufsize;
  size_t bufhead;
  size_t buflen;
};

SNAPPY_FILE snopen(const char* name, const char* mode) {
  SNAPPY_FILE x;

  x = (SNAPPY_FILE)malloc(sizeof(struct snappy_file) * 1);

  if (!x)
    return NULL;

  x->f = fopen(name, mode);
  if (!x->f) {
    free(x);
    return NULL;
  }

  x->buf = (char*)malloc(BUFSIZE);
  if (!x->buf) {
    fclose(x->f);
    free(x);
    return NULL;
  }

  x->bufsize = BUFSIZE;
  x->bufhead = 0;
  x->buflen  = 0;

  return x;
}

size_t snwrite(SNAPPY_FILE f, void* p, size_t sz) {
  size_t clen;

  clen = snappy_max_compressed_length(sz);

  if (clen > f->bufsize) {
    f->buf = (char*)realloc(f->buf, clen);
    if (!f->buf) {
      fprintf(stderr, "snwrite: Out of memory!\n");
      return 0;
    }

    f->bufsize = clen;
  }

  if (snappy_compress(p, sz, f->buf, &clen) == SNAPPY_OK) {
    if (fwrite(&clen, sizeof(clen), 1, f->f) != 1)
      return 0;

    if (fwrite(f->buf, 1, clen, f->f) < clen)
      return 0;

    return sz;
  } else {
    return 0;
  }
}

static size_t snmin(size_t a, size_t b) { return a > b ? b : a; }

size_t snread(SNAPPY_FILE f, void* p, size_t sz) {
  size_t handled = 0;
  size_t read;
  size_t clen, unclen;

  // is there uncompressed data in the buffer?
  assert(f->buflen >= f->bufhead);

  if (f->buflen - f->bufhead) {
    handled = snmin(sz, (f->buflen - f->bufhead));
    memcpy(p, f->buf + f->bufhead, handled);
    sz -= handled;
    p += handled;
    f->bufhead += handled;
  }

  while (sz > 0) {
    assert(f->bufhead == f->buflen);

    f->bufhead = 0;
    f->buflen  = 0;

    if (fread(&clen, sizeof(clen), 1, f->f) != 1) {
      fprintf(stderr, "Failed to read clen (errno: %d, %d, %d)\n", errno,
              ferror(f->f), feof(f->f));
      return handled;
    }

    char* cbuf = (char*)malloc(clen * 1);
    if (!cbuf) {
      fprintf(stderr, "Failed to allocate buffer clen\n");
      return handled;
    }

    if (fread(cbuf, 1, clen, f->f) < clen) {
      fprintf(stderr, "Failed to read complete cbuf of length %u\n", clen);
      // almost certainly an error!
      free(cbuf);
      return handled;
    }

    if (snappy_uncompressed_length(cbuf, clen, &unclen) != SNAPPY_OK) {
      fprintf(stderr, "Failed to decompress uncompressed length %u\n", clen);
      return handled;
    }

    if (unclen > f->bufsize) {
      f->buf = realloc(f->buf, unclen);
      if (!f->buf) {
        fprintf(stderr, "snread: Out of memory!\n");
        return handled;
      }
      f->bufsize = unclen;
    }

    if (snappy_uncompress(cbuf, clen, f->buf, &unclen) != SNAPPY_OK) {
      fprintf(stderr, "Failed to decompress cbuf of length %u\n", clen);
      return handled;
    }
    free(cbuf);

    f->buflen = unclen;

    size_t tocopy;
    tocopy = snmin(sz, f->buflen);

    memcpy(p, f->buf, tocopy);
    sz -= tocopy;
    p += tocopy;
    f->bufhead += tocopy;
    handled += tocopy;
  }

  return handled;
}

int sneof(SNAPPY_FILE f) {
  assert(f);
  return feof(f->f);
}

void snclose(SNAPPY_FILE f) {
  fclose(f->f);
  free(f->buf);
  free(f);
}
