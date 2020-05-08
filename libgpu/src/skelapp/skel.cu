/* -*- mode: c++ -*- */

#include <cuda.h>
#include <cstdio>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>

#include "gg.h"
#include "Timer.h"

extern void gg_main(CSRGraphTy &, CSRGraphTy &);
extern void output(CSRGraphTy &, const char *output_file);
extern const char *GGC_OPTIONS;

int QUIET = 0;
char *INPUT, *OUTPUT;
extern int SKELAPP_RETVAL;
extern unsigned long DISCOUNT_TIME_NS;

unsigned long DISCOUNT_TIME_NS = 0;
int SKELAPP_RETVAL = 0;

int CUDA_DEVICE = 0;

//mgpu::ContextPtr mgc;

extern const char *prog_opts;
extern const char *prog_usage;
extern const char *prog_args_usage;
extern void process_prog_opt(char optchar, char *optarg);
extern int process_prog_arg(int argc, char *argv[], int arg_start);

__global__ void initialize_skel_kernel() {
}

void kernel_sizing(CSRGraphTy & g, dim3 &blocks, dim3 &threads) {
  threads.x = 256;
  threads.y = threads.z = 1;

  blocks.x = ggc_get_nSM() * 8;
  blocks.y = blocks.z = 1;
}

int load_graph_and_run_kernel(char *graph_file) {
  CSRGraphTy g, gg;
  
  ggc::Timer k("gg_main");
  fprintf(stderr, "OPTIONS: %s\n", GGC_OPTIONS);
  g.read(graph_file);

  g.copy_to_gpu(gg);

  int *d;
  check_cuda(cudaMalloc(&d, sizeof(int) * 1));
  //check_cuda(cudaFree(d));

  //initialize_skel_kernel<<<1,1>>>();

  k.start();
  gg_main(g, gg);
  check_cuda(cudaDeviceSynchronize());
  k.stop();
  k.print();
  fprintf(stderr, "Total time: %llu ms\n", k.duration_ms());
  fprintf(stderr, "Total time: %llu ns\n", k.duration());

  if(DISCOUNT_TIME_NS > 0) {
    fprintf(stderr, "Total time (discounted): %llu ns\n", k.duration() - DISCOUNT_TIME_NS);
  }

  gg.copy_to_cpu(g);

  if(!QUIET)
    output(g, OUTPUT);

  return SKELAPP_RETVAL;
}

void usage(int argc, char *argv[]) 
{
  if(strlen(prog_usage)) 
    fprintf(stderr, "usage: %s [-q] [-g gpunum] [-o output-file] %s graph-file \n %s\n", argv[0], prog_usage, prog_args_usage);
  else
    fprintf(stderr, "usage: %s [-q] [-g gpunum] [-o output-file] graph-file %s\n", argv[0], prog_args_usage);
}

void parse_args(int argc, char *argv[]) 
{
  int c;
  const char *skel_opts = "g:qo:";
  char *opts;
  int len = 0;
  
  len = strlen(skel_opts) + strlen(prog_opts) + 1;
  opts = (char *) calloc(1, len);
  strcat(strcat(opts, skel_opts), prog_opts);

  while((c = getopt(argc, argv, opts)) != -1) {
    switch(c) 
      {
      case 'q':
	QUIET = 1;
	break;
      case 'o':
	OUTPUT = optarg; //TODO: copy?
	break;
      case 'g':
	char *end;
	errno = 0;
	CUDA_DEVICE = strtol(optarg, &end, 10);
	if(errno != 0 || *end != '\0') {
	  fprintf(stderr, "Invalid GPU device '%s'. An integer must be specified.\n", optarg);
	  exit(EXIT_FAILURE);
	}
	break;
      case '?':
	usage(argc, argv);
	exit(EXIT_FAILURE);
      default:
	process_prog_opt(c, optarg);
	break;
    }
  }

  if(optind < argc) {
    INPUT = argv[optind];
    if(!process_prog_arg(argc, argv, optind + 1)) {
      usage(argc, argv);
      exit(EXIT_FAILURE);
    }
  }
  else {
    usage(argc, argv);
    exit(EXIT_FAILURE);      
  }
}

int main(int argc, char *argv[]) {
  if(argc == 1) {
    usage(argc, argv);
    exit(1);
  }

  parse_args(argc, argv);
  ggc_set_gpu_device(CUDA_DEVICE);
  //mgc = mgpu::CreateCudaDevice(CUDA_DEVICE);
  //printf("Using GPU: %s\n", mgc->DeviceString().c_str());
  int r = load_graph_and_run_kernel(INPUT);
  return r;
}
