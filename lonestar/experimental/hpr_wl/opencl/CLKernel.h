/*
 * CLKernel.h
 *
 *  Created on: Jun 27, 2015
 *      Author: rashid
 */

#ifndef GDIST_EXP_APPS_HPR_CL_CLKERNEL_H_
#define GDIST_EXP_APPS_HPR_CL_CLKERNEL_H_

namespace galois{
namespace opencl{
extern CLEnvironment cl_env;
struct CL_Kernel_Helper{
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

   static void build_program_source(const char * file_name, const char * _flags = "") {
      cl_int err;
      std::string compiler_flags(_flags);
//      compiler_flags += device_id->get_platrform()->get_cl_compiler_flags();
      compiler_flags += "-I. ";
//      compiler_flags += build_args;
//      fprintf(stderr, "Compiling device : %s, flags : %s, filename : %s \n", device_id->name().c_str(), compiler_flags.c_str(), file_name);
   #if _ALTERA_FPGA_USE_
         {
            FILE *fh;
            std::string compiled_file_name(file_name);
            compiled_file_name.replace(compiled_file_name.length()-3, compiled_file_name.length(),".aocx");
            fprintf(stderr, "About to load binary file :: %s \n", compiled_file_name.c_str());
            if (!(fh = fopen(compiled_file_name.c_str(), "rb")))
               return;
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

      char * src = load_program_source(file_name);
      if (src == NULL || strlen(src) <= 0)
      printf("Empty CL file!!!\n");
      /*std::cout<<"================KERNEL===============\n";
       std::cout<<src<<"\n";
       std::flush(std::cout);*/
      cl_env.m_program = clCreateProgramWithSource(cl_env.m_context, 1, (const char **) &src, NULL, &err);
      CHECK_ERROR_NULL(cl_env.m_program, "clCreateProgramWithSource");
      CHECK_CL_ERROR(err, "clCreateProgramWithSource");
      //check_command_queue(queue);
      const cl_device_id dev_id = cl_env.m_device_id;
      err = clBuildProgram(cl_env.m_program, 1, &dev_id, compiler_flags.c_str(), NULL, NULL);
      if (err != CL_SUCCESS) {
         size_t len = 0;
         cl_int build_status;
         cl_int err2 = clGetProgramBuildInfo(cl_env.m_program,cl_env.m_device_id, CL_PROGRAM_BUILD_STATUS, 0, &build_status, NULL);
   //      CHECK_CL_ERROR(err2, "Failed to build program executable from source");
         cl_int err3 = clGetProgramBuildInfo(cl_env.m_program,cl_env.m_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
         char *buffer = new char[len];
         fprintf(stderr, "Build log length:: %zd\n", len);
         err3 = clGetProgramBuildInfo(cl_env.m_program, cl_env.m_device_id, CL_PROGRAM_BUILD_LOG, len, buffer, &len);
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
         err = clGetProgramInfo(cl_env.m_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_sz, NULL);

         // Read binary (PTX file) to memory buffer
         unsigned char *bin = (unsigned char *) malloc(bin_sz);
         err = clGetProgramInfo(cl_env.m_program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
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
   }
   /**********************************************************************
    *
    *
    **********************************************************************/

   static void build_string_source(const char * src, const char * flags = "") {
      std::string compiler_flags(flags);
//      compiler_flags += device_id->get_platrform()->get_cl_compiler_flags();
//      compiler_flags += " ";
//      compiler_flags += build_args;
      compiler_flags += "-I.";
   //   fprintf(stderr, "Compiling with flags :%s \n", compiler_flags.c_str());
      cl_int err;
      // Create the compute program from the source buffer                   [6]
      cl_env.m_program = clCreateProgramWithSource(cl_env.m_context, 1, (const char **) &src, NULL, &err);
      CHECK_ERROR_NULL(cl_env.m_program, "clCreateProgramWithSource");
      // Build the program executable                                        [7]
      err = clBuildProgram(cl_env.m_program, 0, NULL, compiler_flags.c_str(), NULL, NULL);
      if (err != CL_SUCCESS) {
         size_t len;
         char buffer[10 * 2048];
         clGetProgramBuildInfo(cl_env.m_program, cl_env.m_device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
         std::cout << "\n====Kernel build log====\n" << buffer << "\n=====END LOG=====\n";
      }
      CHECK_CL_ERROR(err, "Failed to build program executable from string");
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   static cl_kernel load_kernel_string(const char * src, const char * kernel_method_name) {
      build_string_source(src, "");
      int err;
      cl_kernel kernel = clCreateKernel(cl_env.m_program, kernel_method_name, &err);
      //      CHECK_CL_ERROR(err,"Error: clCreateKernel\n");
      return kernel;
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   static cl_kernel load_kernel(const char * kernel_file_name, const char * kernel_method_name) {
   //   std::cout << "Loading kernel ... \"" << kernel_file_name << "\", method name " << kernel_method_name << "\n";
      build_program_source(kernel_file_name, "");
      int err;
      cl_kernel kernel = clCreateKernel(cl_env.m_program, kernel_method_name, &err);
      CHECK_CL_ERROR(err, "galois::opencl::OpenCL_Setup::get_default_device()Error: Failed to build kernel in clCreateKernel.");
      return kernel;
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   static size_t workgroup_size(cl_kernel kernel) {
   #ifdef _ALTERA_EMULATOR_USE_
      return 4096;
   #else
      size_t work_group_size;
      CHECK_CL_ERROR(clGetKernelWorkGroupInfo(kernel, cl_env.m_device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL), "Error: Failed to get work-group size for kernel.");
      return work_group_size;
   #endif
   }
   /**********************************************************************
    *
    *
    **********************************************************************/
   static size_t local_memory_size() {
      cl_ulong local_mem;
      CHECK_CL_ERROR(clGetDeviceInfo(cl_env.m_device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem, NULL), "Local mem size");
      return local_mem;
   }
   static cl_ulong get_device_shared_memory(){
      cl_ulong ret;
        CHECK_CL_ERROR(clGetDeviceInfo(cl_env.m_device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &ret, NULL), "Local mem size");
        return ret;
   }
   static cl_uint get_device_eu(){
      cl_uint num_eus;
         CHECK_CL_ERROR(clGetDeviceInfo(cl_env.m_device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_eus, NULL), "clGetDeviceInfo");
         return num_eus;
   }
};

struct CL_Kernel {
   cl_kernel kernel;
   size_t global, local;
   double total_time;
   cl_event event;
   char name[256];
   /*************************************************************************
    *
    *************************************************************************/
   CL_Kernel() : total_time(0), event(0) {
      global = local = 0;
      kernel = 0;
   }
   /*************************************************************************
    *
    *************************************************************************/
   CL_Kernel(const char * filename, const char * kernel_name) :
         total_time(0), event(0) {
      init(filename, kernel_name);
   }
   /*************************************************************************
    *
    *************************************************************************/
   ~CL_Kernel() {
      if (kernel) {
         clReleaseKernel(kernel);
      }
   }
   /*************************************************************************
    *
    *************************************************************************/
   void init(const char * filename, const char * kernel_name) {
      strcpy(name, kernel_name);
      global = local = 0;   //galois::opencl::OpenCL_Setup::get_default_device();
      kernel = galois::opencl::CL_Kernel_Helper::load_kernel(filename, kernel_name);
   }
   /*************************************************************************
    *
    *************************************************************************/
   void init_string(const char * src, const char * kernel_name) {
      strcpy(name, kernel_name);
      global = local = 0;
      kernel = galois::opencl::CL_Kernel_Helper::load_kernel_string(src, kernel_name);
   }
   /*************************************************************************
    *
    *************************************************************************/
   void set_work_size(size_t num_items) {
      local = galois::opencl::CL_Kernel_Helper::workgroup_size(kernel);
      global = (size_t) (ceil(num_items / ((double) local)) * local);
   }
   /*************************************************************************
    *
    *************************************************************************/
   void set_work_size(size_t num_items, size_t local_size) {
      local = local_size;
      global = (size_t) (ceil(num_items / ((double) local)) * local);
   }
   /*************************************************************************
    *
    *************************************************************************/
   size_t get_default_workgroup_size() {
      return 512;//galois::opencl::OpenCL_Setup::workgroup_size(kernel, device);
   }
   /*************************************************************************
    *
    *************************************************************************/
   cl_ulong get_shared_memory_usage() {
      cl_ulong mem_usage;
      CHECK_CL_ERROR(clGetKernelWorkGroupInfo(kernel, cl_env.m_device_id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_usage, NULL),
            "Error: Failed to get shared memory usage for kernel.");
      return mem_usage;
   }
   /*************************************************************************
    *
    *************************************************************************/
   void set_arg(unsigned int index, size_t sz, const void * val) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, index, sz, val), "Arg, compact is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   void set_arg_shmem(unsigned int index, size_t sz) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, index, sz, nullptr), "Arg, shared-mem is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   void set_arg_max_shmem(unsigned int index) {
      size_t sz = CL_Kernel_Helper::get_device_shared_memory() - get_shared_memory_usage();
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, index, sz, nullptr), "Arg, max-shared-mem is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   size_t get_avail_shmem() {
      size_t sz = CL_Kernel_Helper::get_device_shared_memory() - get_shared_memory_usage();
      return sz;
   }
   /*************************************************************************
    *
    *************************************************************************/
   template<typename T>
   void set_arg(unsigned int index, const T &val) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, index, sizeof(cl_mem), &val->device_ptr()), "Arg, compact is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   template<typename T>
   void set_arg_list(const T &val) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &val->device_ptr()), "Arg-1, is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   template<typename T1, typename T2>
   void set_arg_list(const T1 &val1, const T2 &val2) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()), "Arg-1/2, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()), "Arg-2/2, is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   template<typename T1, typename T2, typename T3>
   void set_arg_list(const T1 &val1, const T2 &val2, const T3 &val3) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()), "Arg-1/3, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()), "Arg-2/3, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()), "Arg-3/3, is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   template<typename T1, typename T2, typename T3, typename T4>
   void set_arg_list(const T1 &val1, const T2 &val2, const T3 &val3, const T4 &val4) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()), "Arg-1/4, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()), "Arg-2/4, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()), "Arg-3/4, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()), "Arg-4/4, is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   template<typename T1, typename T2, typename T3, typename T4, typename T5>
   void set_arg_list(const T1 &val1, const T2 &val2, const T3 &val3, const T4 &val4, const T5 &val5) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()), "Arg-1/5, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()), "Arg-2/5, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()), "Arg-3/5, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()), "Arg-4/5, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5->device_ptr()), "Arg-5/5, is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
   void set_arg_list(const T1 &val1, const T2 &val2, const T3 &val3, const T4 &val4, const T5 &val5, const T6 &val6) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()), "Arg-1/6, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()), "Arg-2/6, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()), "Arg-3/6, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()), "Arg-4/6, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5->device_ptr()), "Arg-5/6, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 5, sizeof(cl_mem), &val6->device_ptr()), "Arg-6/6, is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
   void set_arg_list(const T1 &val1, const T2 &val2, const T3 &val3, const T4 &val4, const T5 &val5, const T6 &val6, const T7 & val7) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()), "Arg-1/7, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()), "Arg-2/7, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()), "Arg-3/7, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()), "Arg-4/7, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5->device_ptr()), "Arg-5/7, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 5, sizeof(cl_mem), &val6->device_ptr()), "Arg-6/7, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 6, sizeof(cl_mem), &val7->device_ptr()), "Arg-7/7, is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
   void set_arg_list(const T1 &val1, const T2 &val2, const T3 &val3, const T4 &val4, const T5 &val5, const T6 &val6, const T7 & val7, const T8 & val8) {
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()), "Arg-1/8, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()), "Arg-2/8, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()), "Arg-3/8, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()), "Arg-4/8, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5->device_ptr()), "Arg-5/8, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 5, sizeof(cl_mem), &val6->device_ptr()), "Arg-6/8, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 6, sizeof(cl_mem), &val7->device_ptr()), "Arg-7/8, is NOT set!");
      galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, 7, sizeof(cl_mem), &val8->device_ptr()), "Arg-8/8, is NOT set!");
   }
   /*************************************************************************
    *
    *************************************************************************/
   double get_kernel_time() {
      return total_time;
   }
   /*************************************************************************
    *
    *************************************************************************/
   void set_max_threads() {
      set_work_size(local * galois::opencl::CL_Kernel_Helper::get_device_eu());
   }
   /*************************************************************************
    *
    *************************************************************************/
   void operator()() {
      galois::opencl::CHECK_CL_ERROR(clEnqueueNDRangeKernel(cl_env.m_command_queue, kernel, 1, nullptr, &global, &local, 0, nullptr, &event), name);
   }
   /*************************************************************************
    *
    *************************************************************************/
   void operator()(cl_event & e) {
      galois::opencl::CHECK_CL_ERROR(clEnqueueNDRangeKernel(cl_env.m_command_queue, kernel, 1, NULL, &global, &local, 0, NULL, &e), name);
   }
   /*************************************************************************
    *
    *************************************************************************/
   void wait() {
      galois::opencl::CHECK_CL_ERROR(clWaitForEvents(1, &event), "Error in waiting for kernel.");
   }
   /*************************************************************************
    *
    *************************************************************************/
   void operator()(size_t num_events, cl_event * events) {
      galois::opencl::CHECK_CL_ERROR(clEnqueueNDRangeKernel(cl_env.m_command_queue, kernel, 1, NULL, &global, &local, num_events, events, &event), name);
   }
   /*************************************************************************
    *
    *************************************************************************/
   float last_exec_time() {
#ifdef _GOPT_CL_ENABLE_PROFILING_
      cl_ulong start_time, end_time;
      galois::opencl::CHECK_CL_ERROR(clWaitForEvents(1, &event), "Waiting for kernel");
      galois::opencl::CHECK_CL_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL), "Kernel start time failed.");
      galois::opencl::CHECK_CL_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL), "Kernel end time failed.");
      return (end_time - start_time) / (float) 1.0e9f;
#else
      return 0.0f;
#endif
   }
   /*************************************************************************
    *
    *************************************************************************/
};
}
}
#endif /* GDIST_EXP_APPS_HPR_CL_CLKERNEL_H_ */
