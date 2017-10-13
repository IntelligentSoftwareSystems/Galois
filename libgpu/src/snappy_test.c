#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "snfile.h"

void compress(char *in, char *out) {
  FILE *i;
  SNAPPY_FILE s;
  char *buf;
  size_t N = 1024*1024*100;
  size_t R = 0;

  buf = (char *) malloc(N);
  assert(buf);

  i = fopen(in, "r");
  s = snopen(out, "w");

  assert(i);

  while(!feof(i)) {
    R = fread(buf, 1, N, i);
    
    if(R > 0) {
      snwrite(s, buf, R);
    } else {
      fprintf(stderr, "Error?\n");
    }
  }

  fclose(i);
  snclose(s);  
}

void decompress(char *in, char *out) {
  FILE *o;
  SNAPPY_FILE s;
  char *buf;
  size_t N = 1024;
  size_t R = 0;

  buf = (char *) malloc(N);
  assert(buf);

  s = snopen(in, "r");
  o = fopen(out, "w");

  assert(s);
  assert(o);

  while(!sneof(s)) {
    R = snread(s, buf, N);
    if(R > 0) {
      if(fwrite(buf, 1, R, o) < R) 
	{
	  fprintf(stderr, "Error writing\n");
	  exit(1);
	}
    } else {
      if(!sneof(s)) 
	fprintf(stderr, "Error?\n");
      break;
    }
  }

  snclose(s);  
  fclose(o);
}

int main(int argc, char *argv[]) {
  if(argc != 4) {
    fprintf(stderr, "Usage: %s cmd input output\n", argv[0]);
    exit(1);
  }

  char *cmd = argv[1];
  char *inp = argv[2];
  char *out = argv[3];

  if(strcmp(cmd, "compress") == 0) {
    compress(inp, out);
  } else {
    decompress(inp, out);
  }  
}
