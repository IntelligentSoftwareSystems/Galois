/**
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Rashid Kaleem<rashid.kaleem@gmail.com>
 */
#include <fstream>
#include <string.h>

#ifndef GALOISGPU_OCL_OPENCL_SETUP_H_
#define GALOISGPU_OCL_OPENCL_SETUP_H_

namespace galois {
namespace opencl {
struct CLContext {
   bool initialized;
   std::vector<CL_Platform*> platforms;
   int GOPT_CL_DEFAULT_PLATFORM_ID;
   int GOPT_CL_DEFAULT_DEVICE_ID;
   std::string build_args;
   cl_program program;
   ///////////////////////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////
   void append_build_args(const std::string & s) {
//      fprintf(stderr, "appending : %s \n", s.c_str());
      build_args.append(s);
   }
   void clear_build_args() {
      build_args.clear();
   }
   static inline char *load_program_source(const char *filename) {
      FILE *fh;
      struct stat statbuf;
      char *source;
      if (!(fh = fopen(filename, "rb")))
         return NULL;
      stat(filename, &statbuf);
      source = (char *) malloc(statbuf.st_size + 1);
      fread(source, statbuf.st_size, 1, fh);
      source[statbuf.st_size] = 0;
      return source;
   }
   static inline char *load_program_source(const char *filename, size_t * sz) {
      FILE *fh;
      struct stat statbuf;
      char *source;
      if (!(fh = fopen(filename, "rb")))
         return NULL;
      stat(filename, &statbuf);
      source = (char *) malloc(statbuf.st_size + 1);
      fread(source, statbuf.st_size, 1, fh);
      *sz = statbuf.st_size;
      source[statbuf.st_size] = 0;
      return source;
   }

   /**********************************************************************
    ///////////////////////////////////////////////////////////////////////////
    //device_type can be {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_ACCELERATOR}
    *
    *
    **********************************************************************/
   void scan_system(void) {
      std::cerr << "===============Beginning system scan========================\n";
      cl_platform_id l_platforms[4];
      cl_uint num_platforms;
      CHECK_CL_ERROR(clGetPlatformIDs(1, l_platforms, &num_platforms), "clGetPlatformIDs ");
      for (unsigned int curr_plat = 0; curr_plat < num_platforms; ++curr_plat) { //Each platform
         platforms.push_back(new CL_Platform(l_platforms[curr_plat]));
      } //For each platform
      std::cerr << "===============End system scan========================\n";
      std::fprintf(stderr, "Default device::");
      get_default_device()->print();
      //if you are using a single device and would like to release all
      //the resources for devices.
      if (false) {
         for (unsigned int c = 0; c < num_platforms; ++c) {
            for (unsigned int d = 0; d < platforms[c]->devices.size(); ++d) {
               if (c != (unsigned int) (GOPT_CL_DEFAULT_PLATFORM_ID) && d != (unsigned int) (GOPT_CL_DEFAULT_DEVICE_ID)) {
                  delete platforms[c]->devices[d];
               }
            }
            if (c != (unsigned int) (GOPT_CL_DEFAULT_PLATFORM_ID)) {
               delete platforms[c];
            }
         }
      }

      std::flush(std::cout);
   } //End scan_system
   /**********************************************************************
    *
    *
    **********************************************************************/
   int initialize(void) {
      if (initialized == true) {
//         std::cout << "Double initializing of OpenCL, some items may have different contexts!";
         return 0;
      }
      GOPT_CL_DEFAULT_PLATFORM_ID=0;
      GOPT_CL_DEFAULT_DEVICE_ID=0;
      GOPT_CL_DEFAULT_DEVICE_ID=galois::runtime::NetworkInterface::ID;
      setenv("CUDA_CACHE_DISABLE", "1", 1);
      {
         std::ifstream file("device_default.config");
         if (file.good()) {
            int platform, device;
            file >> platform;
            file >> device;
            GOPT_CL_DEFAULT_DEVICE_ID = device;
            GOPT_CL_DEFAULT_PLATFORM_ID = platform;
         }
      }
      scan_system();
      fprintf(stderr, "Initializing OpenCL! :: ");
      initialized = true;
      return 0;
   }
   cl_device_id get_default_device_id() {
      return get_device(GOPT_CL_DEFAULT_PLATFORM_ID, GOPT_CL_DEFAULT_DEVICE_ID)->id();
   }
   CL_Device * get_default_device() {
      return get_device(GOPT_CL_DEFAULT_PLATFORM_ID, GOPT_CL_DEFAULT_DEVICE_ID);
   }
   CL_Platform * get_default_platform() {
      return platforms[GOPT_CL_DEFAULT_PLATFORM_ID];
   }
   CL_Platform * get_platform(int id) {
      return platforms[id];
   }
   cl_device_id get_device_id(int platform, int dev) {
      assert(platform < (int) platforms.size() && dev < (int) platforms[platform]->devices.size());
      return platforms[platform]->devices[dev]->id();
   }
   CL_Device* get_device(int platform, int dev) {
      assert(platform < (int) platforms.size() && dev < (int) platforms[platform]->devices.size());
      return platforms[platform]->devices[dev];
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   CLContext(){
      initialize();
   }
   ~CLContext(){
      cleanup();
   }
   void cleanup() {
      //TODO RK : cleaup all devices?
      for(size_t i=0; i<platforms.size(); ++i){
         delete platforms[i];
      }
      platforms.clear();
   }
   /**********************************************************************
    *
    *
    **********************************************************************/

   void build_string_source(const char * src, const char * flags, CL_Device* device_id) {
      std::string compiler_flags(flags);
      compiler_flags += device_id->get_platform()->get_cl_compiler_flags();
      compiler_flags += " ";
      compiler_flags += build_args;
//   fprintf(stderr, "Compiling with flags :%s \n", compiler_flags.c_str());
      cl_int err;
      // Create the compute program from the source buffer                   [6]
      program = clCreateProgramWithSource(device_id->context(), 1, (const char **) &src, NULL, &err);
      CHECK_ERROR_NULL(program, "clCreateProgramWithSource");
      // Build the program executable                                        [7]
      err = clBuildProgram(program, 0, NULL, compiler_flags.c_str(), NULL, NULL);
      if (err != CL_SUCCESS) {
         size_t len;
         char buffer[10 * 2048];
         clGetProgramBuildInfo(program, device_id->id(), CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
         std::cout << "\n====Kernel build log====\n" << buffer << "\n=====END LOG=====\n";
      }
      CHECK_CL_ERROR(err, "Failed to build program executable from string");
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
#if 0
   static void build_program_binary(const char * file_name, const char * _flags = "", CL_Device * device_id = get_default_device()) {
      //   program = clCreateProgramWithSource(device_id->context(), 1, (const char **) &src, NULL, &err);
      {
         FILE *fh;
         if (!(fh = fopen(file_name, "rb")))
         return;
         fseek(fh, 0, SEEK_END);
         size_t len = ftell(fh);
         unsigned char * source = (unsigned char *) malloc(len);
         rewind(fh);
         fread(source, len, 1, fh);
         cl_device_id dev_id = device_id->id();
         program = clCreateProgramWithBinary(device_id->context(), 1, &dev_id, &len, (const unsigned char **) &src, NULL, &err);
         CHECK_ERROR_NULL(program, "clCreateProgramWithSource");
         fprintf(stderr, "Loaded program successfully [Len=%d]....\n", len);
      }
   }
#endif
   /**********************************************************************
    *
    *
    **********************************************************************/
   void build_program_source(const char * file_name, const char * _flags, CL_Device * device_id) {
      cl_int err;
      std::string compiler_flags(_flags);
      compiler_flags += device_id->get_platform()->get_cl_compiler_flags();
      compiler_flags += " ";
      compiler_flags += build_args;
      fprintf(stderr, "Compiling device : %s, flags : %s, filename : %s \n", device_id->name().c_str(), compiler_flags.c_str(), file_name);
#ifdef _ALTERA_FPGA_USE_
      {
         FILE *fh;
         std::string compiled_file_name(file_name);
         compiled_file_name.replace(compiled_file_name.length()-3, compiled_file_name.length(),".aocx");
         fprintf(stderr, "About to load binary file :: %s \n", compiled_file_name.c_str());
         if (!(fh = fopen(compiled_file_name.c_str(), "rb"))){
            fprintf(stderr, "ERROR - File [%s] not found\n", compiled_file_name.c_str());
            return;
         }
         fseek(fh, 0, SEEK_END);
         size_t len = ftell(fh);
         unsigned char * source = (unsigned char *) malloc(len);
         rewind(fh);
         fread(source, len, 1, fh);
         cl_device_id dev_id = device_id->id();
         program = clCreateProgramWithBinary(device_id->context(), 1, &dev_id, &len, (const unsigned char **) &source, NULL, &err);
         CHECK_ERROR_NULL(program, "clCreateProgramWithSource");
         fprintf(stderr, "Loaded program successfully [Len=%d]....\n", len);
      }
#else
#if 0
      {
         FILE *fh;
         char compiled_file_name[1024];
         sprintf(compiled_file_name, "%s_0.ptx", file_name);
         fprintf(stderr, "About to load nv-ptx file :: %s \n", compiled_file_name);
         if (!(fh = fopen(compiled_file_name, "rb")))
         return;
         fseek(fh, 0, SEEK_END);
         size_t len = ftell(fh);
         unsigned char * source = (unsigned char *) malloc(len);
         rewind(fh);
         fread(source, len, 1, fh);
         fprintf(stderr, "Loaded program from disk [Len=%d]....\n", len);
         cl_device_id dev_id = device_id->id();
         program = clCreateProgramWithBinary(device_id->context(), 1, &dev_id, &len, (const unsigned char **) &source, NULL, &err);
         CHECK_ERROR_NULL(program, "clCreateProgramWithBinary");
         fprintf(stderr, "Loaded program successfully [Len=%d]....\n", len);
      }

#else
      char * src = load_program_source(file_name);
      if (src == NULL || strlen(src) <= 0)
         printf("Empty CL file!!!\n");
      /*std::cout<<"================KERNEL===============\n";
       std::cout<<src<<"\n";
       std::flush(std::cout);*/
      program = clCreateProgramWithSource(device_id->context(), 1, (const char **) &src, NULL, &err);
      CHECK_ERROR_NULL(program, "clCreateProgramWithSource");
      CHECK_CL_ERROR(err, "clCreateProgramWithSource");
      //check_command_queue(queue);
      const cl_device_id dev_id = device_id->id();
      err = clBuildProgram(program, 1, &dev_id, compiler_flags.c_str(), NULL, NULL);
      if (err != CL_SUCCESS) {
         size_t len = 0;
         cl_int build_status;
         cl_int err2 = clGetProgramBuildInfo(program, device_id->id(), CL_PROGRAM_BUILD_STATUS, 0, &build_status, NULL);
//      CHECK_CL_ERROR(err2, "Failed to build program executable from source");
         cl_int err3 = clGetProgramBuildInfo(program, device_id->id(), CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
         char *buffer = new char[len];
         fprintf(stderr, "Build log length:: %zd\n", len);
         err3 = clGetProgramBuildInfo(program, device_id->id(), CL_PROGRAM_BUILD_LOG, len, buffer, &len);
         fprintf(stderr, "\n====Kernel build log====\n%s\n=====END LOG=====\n", buffer);
         std::cout << "ERR : (" << err << "), Code: [status]" << err2 << ", [log]" << err3 << ", Length: " << len << "\n";
         CHECK_CL_ERROR(err, "Failed to build program executable from source");
         delete[] buffer;
      }
      {
         //Generates ptx binaries for NVCC, and unknown (as yet) for other platforms.
         static int ptx_file_counter = 0;
         // Query binary (PTX file) size
         size_t bin_sz;
         err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_sz, NULL);

         // Read binary (PTX file) to memory buffer
         unsigned char *bin = (unsigned char *) malloc(bin_sz);
         err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
         // Save PTX to add_vectors_ocl.ptx
         char filename[256];
         sprintf(filename, "%s_%d.ptx", file_name, ptx_file_counter);
         FILE *fp = fopen(filename, "wb");
         fwrite(bin, sizeof(char), bin_sz, fp);
         fclose(fp);
         free(bin);
      }
      free(src);
      CHECK_CL_ERROR(err, "Failed to build program executable from source");
#endif

#endif
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   void check_kernel(cl_kernel kernel, cl_device_id did) {
      cl_uint num_args;
      char name[1024];
      CHECK_CL_ERROR(clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &num_args, NULL), "Error: Failed to get #Args for kernel.");
      CHECK_CL_ERROR(clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(char) * 1024, name, NULL), "Error: Failed to NAME for kernel.");
      std::cout << "Kernel :" << name << " , local size : " << workgroup_size(kernel, did) << "  Args : " << num_args << "\n";
      return;
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   cl_kernel load_kernel_string(const char * src, const char * kernel_method_name, CL_Device * device) {
      build_string_source(src, "", device);
      int err;
      cl_kernel kernel = clCreateKernel(program, kernel_method_name, &err);
//      fprintf(stderr, "\n%s\n%s\n", src, kernel_method_name);
      CHECK_CL_ERROR(err,"Error: clCreateKernel\n");
      return kernel;
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   cl_kernel load_kernel(const char * kernel_file_name, const char * kernel_method_name, CL_Device * device) {
//   std::cout << "Loading kernel ... \"" << kernel_file_name << "\", method name " << kernel_method_name << "\n";
      build_program_source(kernel_file_name, "", device);
      int err;
      cl_kernel kernel = clCreateKernel(program, kernel_method_name, &err);
      CHECK_CL_ERROR(err, "galois::opencl::OpenCL_Setup::get_default_device()Error: Failed to build kernel in clCreateKernel.");
      return kernel;
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   size_t workgroup_size(cl_kernel kernel, CL_Device * device) {
#ifdef _ALTERA_FPGA_USE_
      return 4096;
#else
      size_t work_group_size;
      CHECK_CL_ERROR(clGetKernelWorkGroupInfo(kernel, device->id(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL),
            "Error: Failed to get work-group size for kernel.");
      return work_group_size;
#endif
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   size_t workgroup_size(cl_kernel kernel, cl_device_id id) {
#ifdef _ALTERA_FPGA_USE_
      return 4096;
#else
      size_t work_group_size;
      CHECK_CL_ERROR(clGetKernelWorkGroupInfo(kernel, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL), "Error: Failed to get work-group size for kernel.");
      return work_group_size;
#endif
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   size_t local_memory_size(CL_Device * device) {
      cl_ulong local_mem;
      CHECK_CL_ERROR(clGetDeviceInfo(device->id(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem, NULL), "Local mem size");
      return local_mem;
   }
   /**********************************************************************
    *
    *
    **********************************************************************/

   double print_event(cl_event event) {
      //CL_PROFILING_COMMAND_END
      //CL_PROFILING_COMMAND_START
      //CL_PROFILING_COMMAND_SUBMIT
      //CL_PROFILING_COMMAND_QUEUED
      cl_ulong time_start, time_end;
      clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
      clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_end, NULL);
      cl_double TimeMs = (cl_double) (time_end - time_start) * (cl_double) (1e-06);
      return TimeMs;
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   const char * toString_mem_object(const cl_mem_object_type &t) {
      if (t & CL_MEM_OBJECT_BUFFER)
         std::cout << "CL_MEM_OBJECT_BUFFER ";
      if (t & CL_MEM_OBJECT_IMAGE2D)
         std::cout << "CL_MEM_OBJECT_IMAGE2D ";
      if (t & CL_MEM_OBJECT_IMAGE3D)
         std::cout << "CL_MEM_OBJECT_IMAGE3D ";
      return "";
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   const char * toString_mem_flags(const cl_mem_flags & t) {
      if (t & CL_MEM_READ_WRITE)
         std::cout << "CL_MEM_READ_WRITE ";
      if (t & CL_MEM_WRITE_ONLY)
         std::cout << "CL_MEM_WRITE_ONLY ";
      if (t & CL_MEM_READ_ONLY)
         std::cout << "CL_MEM_READ_ONLY ";
      if (t & CL_MEM_USE_HOST_PTR)
         std::cout << "CL_MEM_USE_HOST_PTR ";
      if (t & CL_MEM_ALLOC_HOST_PTR)
         std::cout << "CL_MEM_ALLOC_HOST_PTR ";
      if (t & CL_MEM_COPY_HOST_PTR)
         std::cout << "CL_MEM_COPY_HOST_PTR ";
      return " ";
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   void check_memory(const cl_mem & ptr) {
      cl_mem_object_type mem_type;
      CHECK_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_TYPE, sizeof(mem_type), &mem_type, 0), "Check memory type failed!");
      cl_mem_flags mflags;
      CHECK_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_FLAGS, sizeof(mflags), &mflags, 0), "Check memory flags failed!");
      size_t size_mem;
      CHECK_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_SIZE, sizeof(size_mem), &size_mem, 0), "Check memory size failed!");
      void * host_ptr;
      CHECK_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_HOST_PTR, sizeof(host_ptr), &host_ptr, 0), "Check memory host-ptr failed!");
      cl_uint map_count;
      CHECK_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_MAP_COUNT, sizeof(map_count), &map_count, 0), "Check memory map-count failed!");
      cl_uint ref_count;
      CHECK_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_REFERENCE_COUNT, sizeof(ref_count), &ref_count, 0), "Check memory ref-count failed!");
      cl_context ctx;
      CHECK_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_CONTEXT, sizeof(ctx), &ctx, 0), "Check memory coontext failed!");
      cl_mem parent;
      CHECK_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_ASSOCIATED_MEMOBJECT, sizeof(cl_mem), &parent, 0), "Check memory parent failed!");
      size_t offset;
      CHECK_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_OFFSET, sizeof(size_t), &offset, 0), "Check memory offset failed!");
      std::cout << "Mem-info :: Type(" << toString_mem_object(mem_type) << "), Flags(";
      toString_mem_flags(mflags);
      std::cout << "= " << mflags << ") Size (" << size_mem << ")";
      std::cout << " Host( " << host_ptr << "), map_count(" << map_count << ") , ref_count (" << ref_count << "), ctx (" << ctx << ")";
      std::cout << " Offset (" << offset << " , Parent? " << parent << "\n";
   }
   /**********************************************************************
    *
    *
    **********************************************************************/

   cl_uint get_device_eu() {
      return galois::opencl::get_device_eu(get_default_device_id());
   }
   /**********************************************************************
    *
    *
    **********************************************************************/

   cl_uint get_device_threads() {
      return galois::opencl::get_device_threads(get_default_device_id());
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   cl_ulong get_device_memory() {
      return galois::opencl::get_device_memory(get_default_device_id());
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   cl_ulong get_max_allocation_size() {
      return galois::opencl::get_max_allocation_size(get_default_device_id());
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   void finish() {
      galois::opencl::CLContext::get_default_device()->finish();

   }   //wait
   /**********************************************************************
    *
    *
    **********************************************************************/
}
;
//struct OpenCLSetup
}//namespace opencl
} // namespace galois

#endif /* GALOISGPU_OCL_OPENCL_SETUP_H_ */
