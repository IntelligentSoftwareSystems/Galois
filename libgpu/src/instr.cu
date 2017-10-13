/* -*- mode: c++ -*- */
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "gg.h"
#include "bmk2.h"
#include <zlib.h>

#ifdef USE_SNAPPY
#include "snfile.h"
#endif

enum compformat {
  UNCOMPRESSED = 0,
  GZIP = 1,
  SNAPPY = 2
};

struct trace_file {
  int format;
  union {
    FILE *f;
    gzFile z;
#ifdef USE_SNAPPY
    SNAPPY_FILE s;
#endif    
  };
};

static const char *saved_uniqid;

TRACE trace_open(const char *name, const char *mode) {
  trace_file *t;
  int use_compress = 1;
  const char *c;

  t = (trace_file *) malloc(sizeof(trace_file) * 1);

  if(!t) {
    fprintf(stderr, "(Internal) Unable to allocate memory for TRACE '%s' (mode: %s)\n", name, mode);
    exit(1);
  }

  if(c = getenv("INSTR_COMPRESS")) {
    use_compress = atoi(c);
    fprintf(stderr, "Instr Compression enabled: %d\n", use_compress);
  }

  if(!use_compress) {
    t->format = UNCOMPRESSED;
    t->f = fopen(name, mode);
  
    if(!t->f) {
      fprintf(stderr, "Unable to open trace data file '%s' (mode: %s)\n", name, mode);
      exit(1);
    }
  } else {
#ifdef USE_SNAPPY
    t->format = SNAPPY;
    t->s = snopen(name, mode);
    if(!t->s) {
      fprintf(stderr, "Unable to open compressed trace data file '%s' (mode: %s)\n", name, mode);
      exit(1);      
    }
#else
    t->format = GZIP;
    t->z = gzopen(name, mode);
    
    gzbuffer(t->z, 1048576);

    int gzip_level = 3;

    if(c = getenv("INSTR_GZIP_LEVEL")) {
      gzip_level = atoi(c);
      fprintf(stderr, "Using GZIP level: %d\n", gzip_level);
    }

    gzsetparams(t->z, gzip_level, Z_DEFAULT_STRATEGY);

    if(!t->z) {
      fprintf(stderr, "Unable to open compressed trace data file '%s' (mode: %s)\n", name, mode);
      exit(1);
    }
#endif
  }

  return t;
}

void trace_close(TRACE t) {

  if(t->format == UNCOMPRESSED) {
    fclose(t->f);
  } else if (t->format == GZIP) {
    gzclose(t->z);
  } else if (t->format == SNAPPY) {
#ifdef USE_SNAPPY
    snclose(t->s);
#endif 
  }

  free(t);
}


void instr_set_saved_uniqid(const char *id) {
  saved_uniqid = id;
}

void instr_load_uniqid() {
  const int SZ=255;
  static char id[SZ];
  const char *r = NULL;

  r = getenv("INSTR_UNIQID");

  if(r) {
    strncpy(id, r, SZ);
    assert(id[SZ - 1] == '\0');
    instr_set_saved_uniqid(id);
  } else {
    fprintf(stderr, "Unable to read environment variable INSTR_UNIQID\n");
    exit(1);
  }
}

const char *instr_trace_dir() {
  const int SZ=255;
  static char dir[SZ];
  static bool checked;
  const char *r = NULL;
  
  if(!checked) {
    r = getenv("INSTR_TRACE_DIR");
    
    if(r) {
      strncpy(dir, r, SZ);
      assert(dir[SZ - 1] == '\0');
      //TODO: append a "/"?
    } else {    
      dir[0] = '\0';
    }
    checked = true;
  }

  return dir;
}

const char *instr_saved_uniqid() {
  return saved_uniqid;
}

const char *instr_uniqid() {
  const char *runid;
  static char spid[32];
  int ret;

  runid = bmk2_get_runid();
  if(!runid) {
    ret = snprintf(spid, 32, "%d", getpid());
    assert(ret > 0 && ret < 32);
    runid = spid;
  }

  return runid;
}

void instr_write_array(const char *n, 
		      TRACE f, size_t elemsz, size_t nelems, void *p) 
{
  assert(f != NULL);

  if(f->format == UNCOMPRESSED) {
    if(fwrite(&nelems, sizeof(nelems), 1, f->f) != 1) {
      fprintf(stderr, "Error writing size to '%s'\n", n);
      exit(1);
    }

    if(fwrite(p, elemsz, nelems, f->f) != nelems) {
      fprintf(stderr, "Error writing items to '%s'\n", n);
      exit(1);
    }
  } else if(f->format == GZIP) {
    if(gzwrite(f->z, &nelems, sizeof(nelems) * 1) < sizeof(nelems) * 1) {
      fprintf(stderr, "Error writing size to compressed '%s'\n", n);
      exit(1);
    }
    
    if(gzwrite(f->z, p, elemsz * nelems) < elemsz * nelems) {
      fprintf(stderr, "Error writing items to compressed '%s'\n", n);
      exit(1);
    }
  } else if(f->format == SNAPPY) {
#ifdef USE_SNAPPY
    if(snwrite(f->s, &nelems, sizeof(nelems) * 1) < sizeof(nelems) * 1) {
      fprintf(stderr, "Error writing size to compressed '%s'\n", n);
      exit(1);
    }
    
    if(snwrite(f->s, p, elemsz * nelems) < elemsz * nelems) {
      fprintf(stderr, "Error writing items to compressed '%s'\n", n);
      exit(1);
    }
#endif 
  }
}

#ifdef USE_SNAPPY
SNAPPY_FILE trace_snappy_handle(TRACE f) {
  return f->s;
}
#endif

size_t instr_read_array(const char *n, 
			TRACE f, 
			size_t elemsz, 
			size_t maxnelems, 
			void *p) 
{
  size_t nelems;

  assert(f != NULL);
  if(f->format == UNCOMPRESSED) {
    if(fread(&nelems, sizeof(nelems), 1, f->f) != 1) {
      fprintf(stderr, "Error reading size from '%s'\n", n);
      exit(1);
    }

    if(nelems > maxnelems) {
      fprintf(stderr, "Too many items to read from '%s'\n", n);
      exit(1);
    }

    if(fread(p, elemsz, nelems, f->f) != nelems) {
      fprintf(stderr, "Error reading items from '%s'\n", n);
      exit(1);
    }
  } else if(f->format == GZIP) {
    if(gzread(f->z, &nelems, sizeof(nelems) * 1) < sizeof(nelems) * 1) {
      fprintf(stderr, "Error reading size from compressed '%s'\n", n);
      exit(1);
    }

    if(nelems > maxnelems) {
      fprintf(stderr, "Too many items to read from '%s'\n", n);
      exit(1);
    }

    if(gzread(f->z, p, elemsz * nelems) != elemsz * nelems) {
      fprintf(stderr, "Error reading items from compressed '%s'\n", n);
      exit(1);
    }
  } else if(f->format == SNAPPY) {
#ifdef USE_SNAPPY
    if(snread(f->s, &nelems, sizeof(nelems) * 1) < sizeof(nelems) * 1) {
      fprintf(stderr, "Error reading size from compressed '%s'\n", n);
      exit(1);
    }

    if(nelems > maxnelems) {
      fprintf(stderr, "Too many items to read from '%s'\n", n);
      exit(1);
    }

    if(snread(f->s, p, elemsz * nelems) != elemsz * nelems) {
      fprintf(stderr, "Error reading items from compressed '%s'\n", n);
      exit(1);
    }
#endif 
  }
  return nelems;
}

size_t instr_read_array_gpu(const char *n, 
			    TRACE f, size_t elemsz, size_t maxnelems, 
			    void *gp, void *cp) 
{
  bool allocated = false;

  if(!cp) {
    cp = malloc(elemsz * maxnelems);
    allocated = true;
    assert(cp != NULL);
  }

  size_t nelems = instr_read_array(n, f, elemsz, maxnelems, cp);

  check_cuda(cudaMemcpy(gp, cp, nelems * elemsz, cudaMemcpyHostToDevice));
  
  if(allocated) 
    free(cp);

  return nelems;
}


void instr_write_array_gpu(const char *n, 
			   TRACE f, size_t elemsz, size_t nelems, 
			   void *gp, void *cp) 
{
  bool allocated = false;

  if(!cp) {
    cp = malloc(elemsz * nelems);
    allocated = true;
    assert(cp != NULL);
  }

  check_cuda(cudaMemcpy(cp, gp, nelems * elemsz, cudaMemcpyDeviceToHost));
  
  instr_write_array(n, f, elemsz, nelems, cp);

  if(allocated) 
    free(cp);
}


void instr_save_primitive(const char *name, 
			  const int invocation,
			  const int pos,
			  const char *arg,
			  void *p, size_t sp)
{
  const int SZ=255;
  char fname[SZ];
  int written;

  written = snprintf(fname, SZ, "%s%s.%s.%s.arg", instr_trace_dir(), name, arg, instr_uniqid());
  
  assert(written > 0 && written < SZ);
  
  FILE *o;
  o = fopen(fname, "w+");

  if(o == NULL) {
    fprintf(stderr, "Failed to open '%s'\n", fname);
    exit(1);
  }

  if(fseek(o, invocation * sp, SEEK_SET) == 0) {
    if(fwrite(p, sp, 1, o) != 1) {
      fprintf(stderr, "instr_save_primitive: Write failed!\n");
      exit(1);
    }
  }
  else {
      fprintf(stderr, "instr_save_primitive: fseek failed!\n");
      exit(1);    
  }

  if(invocation == 0) {
    bmk2_log_collect("ggc/kstate", fname);
  }

  fclose(o);
}

void instr_load_primitive(const char *name, 
			  const int invocation,
			  const int pos,
			  const char *arg,
			  void *p, size_t sp)
{
  const int SZ=255;
  char fname[SZ];
  int written;

  written = snprintf(fname, SZ, "%s%s.%s.%s.arg", instr_trace_dir(), name, arg, instr_saved_uniqid());
  
  assert(written > 0 && written < SZ);
  
  FILE *o;
  o = fopen(fname, "r");

  if(o == NULL) {
    fprintf(stderr, "Failed to open '%s'\n", fname);
    exit(1);
  }

  if(fseek(o, invocation * sp, SEEK_SET) == 0) {
    if(fread(p, sp, 1, o) != 1) {
      fprintf(stderr, "instr_load_primitive: Read failed!\n");
      exit(1);
    }
  }
  else {
      fprintf(stderr, "instr_load_primitive: fseek failed!\n");
      exit(1);    
  }

  fclose(o);
}


void instr_save_array_gpu(const char *kernel,
			  const int invocation,
			  const int pos,
			  const char *arg,
			  void *gp,
			  void *cp,
			  size_t sz,
			  size_t num)
{
  const int SZ=255;
  char fname[SZ];
  int written;

  written = snprintf(fname, SZ, "%s%s.%s.%d.%s.arg", instr_trace_dir(), kernel, arg, invocation, instr_uniqid());
  
  assert(written >0 && written < SZ);
  
  TRACE o;
  o = trace_open(fname, "w");

  instr_write_array_gpu(fname, o, sz, num, gp, cp);

  bmk2_log_collect("ggc/kstate", fname);

  trace_close(o);
}

void instr_save_array(const char *kernel,
		      const int invocation,
		      const int pos,
		      const char *arg,
		      void *cp,
		      size_t sz,
		      size_t num)
{
  const int SZ=255;
  char fname[SZ];
  int written;

  written = snprintf(fname, SZ, "%s%s.%s.%d.%s.arg", instr_trace_dir(), kernel, arg, invocation, instr_uniqid());
  
  assert(written >0 && written < SZ);
  
  TRACE o;
  o = trace_open(fname, "w");

  instr_write_array(fname, o, sz, num, cp);

  bmk2_log_collect("ggc/kstate", fname);

  trace_close(o);
}


size_t instr_load_array_gpu(const char *kernel,
			    const int invocation,
			    const int pos,
			    const char *arg,
			    void *gp,
			    void *cp,
			    size_t sz,
			    size_t maxnum)
{
  const int SZ=255;
  char fname[SZ];
  int written;

  written = snprintf(fname, SZ, "%s%s.%s.%d.%s.arg", instr_trace_dir(), kernel, arg, invocation, 
		     instr_saved_uniqid());
  
  assert(written >0 && written < SZ);
  
  TRACE o;
  o = trace_open(fname, "r");

  if(o == NULL) {
    fprintf(stderr, "Failed to open '%s'\n", fname);
    exit(1);
  }

  assert(o != NULL);

  size_t nelems = instr_read_array_gpu(fname, o, sz, maxnum, gp, cp);
  
  trace_close(o);

  return nelems;
}

size_t instr_load_array(const char *kernel,
			const int invocation,
			const int pos,
			const char *arg,
			void *cp,
			size_t sz,
			size_t maxnum)
{
  const int SZ=255;
  char fname[SZ];
  int written;

  written = snprintf(fname, SZ, "%s%s.%s.%d.%s.arg", instr_trace_dir(), kernel, arg, invocation, 
		     instr_saved_uniqid());
  
  assert(written >0 && written < SZ);
  
  TRACE o;
  o = trace_open(fname, "r");

  size_t nelems = instr_read_array(fname, o, sz, maxnum, cp);
  
  trace_close(o);

  return nelems;
}


struct instr_trace_record {
  int ty;
  int depth;
  int index;
};

struct instr_trace {
  FILE *f;  
  struct instr_trace_record *r;
  int records;
  int index;
};


struct instr_trace * instr_trace_file(const char *prefix, int mode) {
  struct instr_trace *it;
  const int SZ=255;
  char fname[SZ];
  int written;
  const char *id;

  it = (struct instr_trace *) malloc(sizeof(instr_trace));
  it->r = NULL;

  if(mode == 0) {
    id = instr_saved_uniqid(); // read
  } else {
    id = instr_uniqid(); //write
  }

  written = snprintf(fname, SZ, "%s%s.%s.trace", instr_trace_dir(), prefix, id);
  assert(written >0 && written < SZ);

  if(mode == 0) 
    it->f = fopen(fname, "r");
  else 
    it->f = fopen(fname, "w");

  if(!it->f) {
    fprintf(stderr, "Failed to open '%s' for %s\n", fname, mode ? "writing" : "reading");
    exit(1);  
  }

  if(mode == 0) {
    instr_load_trace(fname, it);
  } else {
    bmk2_log_collect("ggc/trace", fname);
  }

  return it;
}

void instr_load_trace(const char *n, struct instr_trace *it) {
  assert(it != NULL);
  assert(it->f != NULL);
  assert(it->r == NULL);

  int wr = 0;
  int N = 1024;
  int ty, depth, index;

  it->r = (struct instr_trace_record *) malloc(N*sizeof(instr_trace_record));

  assert(it->r);

  int fd = fileno(it->f);
  assert(fd != -1);

  struct stat s;

  if(fstat(fd, &s) != 0) {
    fprintf(stderr, "Error when stat'ing trace file '%s'\n", n);
    exit(1);
  }

  if(s.st_size > 0) {
    while(!feof(it->f)) {
      if(fscanf(it->f, "%d %d %d\n", &ty, &depth, &index) == 3) {
	it->r[wr].ty = ty;
	it->r[wr].depth = depth;
	it->r[wr].index = index;
	wr++;
	if(wr >= N) {
	  N *= 2;
	  it->r = (struct instr_trace_record *) realloc(it->r, N*sizeof(instr_trace_record));
	  assert(it->r != NULL);
	}
      }
      else {
	fprintf(stderr, "Error when reading trace file '%s'\n", n);
	exit(1);
      }
    }
  }

  it->records = wr;
  it->index = 0;
}

bool instr_match_pipe(struct instr_trace *it, int what, int depth, int index) {
  assert(it != NULL);
  assert(it->f != NULL);
  assert(it->r != NULL);
  
  if(it->index < it->records) {
    struct instr_trace_record *r = &it->r[it->index];
    if (r->ty == what && r->depth == depth && r->index == index) {
      it->index++;
      return true;
    }
  }

  return false;
}

bool debug_match(struct instr_trace *it, bool succeeded, int depth, int index, int what, const char *swhat) {
  if(!succeeded) {
    assert(it != NULL);
    assert(it->f != NULL);
    assert(it->r != NULL);

    if(it->index < it->records) {
      struct instr_trace_record *r = &it->r[it->index];
      
      fprintf(stderr, "Match attempt failed: (ty: %d/%s, depth: %d, index: %d) does not equal stored (ty: %d, depth: %d, index: %d)\n", 
	      what, swhat, depth, index, 
	      r->ty, r->depth, r->index);      
    } else {
      fprintf(stderr, "Match attempt failed: (ty: %d/%s, depth: %d, index: %d) beyond end of stored trace\n", 
	      what, swhat, depth, index);
    }
  }

  return succeeded;
}

bool instr_match_pipe_iterate(struct instr_trace *it, int depth, int index) {
  bool x = instr_match_pipe(it, INSTR_TRACE_ITER, depth, index);

  return debug_match(it, x, depth, index, INSTR_TRACE_ITER, "ITER");
}

bool instr_match_pipe_exit(struct instr_trace *it, int depth, int index) {
  bool x = instr_match_pipe(it, INSTR_TRACE_EXIT, depth, index);

  return debug_match(it, x, depth, index, INSTR_TRACE_EXIT, "EXIT");
}

void instr_pipe_iterate(struct instr_trace *it, int depth, int index) {
  assert(it != NULL);
  assert(it->f != NULL);
  fprintf(it->f, "%d %d %d\n", INSTR_TRACE_ITER, depth, index);
}

void instr_pipe_exit(struct instr_trace *it, int depth, int index) {
  assert(it != NULL);
  assert(it->f != NULL);
  fprintf(it->f, "%d %d %d\n", INSTR_TRACE_EXIT, depth, index);
}
