#pragma once

typedef struct trace_file * TRACE;
struct instr_trace;

void instr_set_saved_uniqid(const char *id);
void instr_load_uniqid();
const char *instr_saved_uniqid();
const char *instr_uniqid();
const char *instr_trace_dir();

TRACE trace_open(const char *name, const char *mode);
void trace_close(TRACE t);

void instr_write_array(const char *n, 
		       TRACE f, size_t elemsz, size_t nelems, void *p);


/* gp is the gpu pointer, cp can be the associated CPU pointer if available */
void instr_write_array_gpu(const char *n, 
			   TRACE f, size_t elemsz, size_t nelems, 
			   void *gp, void *cp);

size_t instr_read_array(const char *n, 
			TRACE f, 
			size_t elemsz, 
			size_t maxnelems, 
			void *p);

size_t instr_read_array_gpu(const char *n, 
			    TRACE f, size_t elemsz, size_t maxnelems, 
			    void *gp, void *cp);

void instr_save_array_gpu(const char *kernel,
			  const int invocation,
			  const int pos,
			  const char *arg,
			  void *gp,
			  void *cp,
			  size_t sz,
			  size_t num);

void instr_save_array(const char *kernel,
		      const int invocation,
		      const int pos,
		      const char *arg,
		      void *cp,
		      size_t sz,
		      size_t num);


void instr_save_primitive(const char *name, 
			  const int invocation,
			  const int pos,
			  const char *arg,
			  void *p, size_t sp);

size_t instr_load_array_gpu(const char *kernel,
			    const int invocation,
			    const int pos,
			    const char *arg,
			    void *gp,
			    void *cp,
			    size_t sz,
			    size_t maxnum);

size_t instr_load_array(const char *kernel,
			const int invocation,
			const int pos,
			const char *arg,
			void *cp,
			size_t sz,
			size_t maxnum);

void instr_load_primitive(const char *name, 
			  const int invocation,
			  const int pos,
			  const char *arg,
			  void *p, size_t sp);

struct instr_trace* instr_trace_file(const char *prefix, int mode);

void instr_pipe_iterate(struct instr_trace *f, int depth, int index);

void instr_pipe_exit(struct instr_trace *f, int depth, int index);

void instr_load_trace(const char *n, struct instr_trace *it);
bool instr_match_pipe(struct instr_trace *it, int what, int depth, int index);
bool instr_match_pipe_iterate(struct instr_trace *it, int depth, int index);
bool instr_match_pipe_exit(struct instr_trace *it, int depth, int index);
void instr_pipe_iterate(struct instr_trace *it, int depth, int index);
void instr_pipe_exit(struct instr_trace *it, int depth, int index);

#ifdef USE_SNAPPY
#include "snfile.h"
SNAPPY_FILE trace_snappy_handle(TRACE f);
#endif


#define INSTR_TRACE_ITER 0
#define INSTR_TRACE_EXIT 1
