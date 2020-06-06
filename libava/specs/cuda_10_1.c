ava_name("Katana CUDA Runtime");
ava_version("10.1.0");
ava_identifier(KATANA_CUDA);
ava_number(1);
ava_cflags(-I/usr/local/cuda-10.1/include -I../headers);
ava_libs(-L/usr/local/cuda-10.1/lib64 -lcudart -lcuda);
ava_export_qualifier();

/**
 * Compile by
 * ./nwcc samples/cudart.nw.c -I /usr/local/cuda-10.0/include -I headers `pkg-config --cflags glib-2.0`
 */

ava_non_transferable_types {
    ava_handle;
}

size_t __args_index_0;
size_t __kernelParams_index_0;

ava_begin_utility;
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fatbinary.h>
#include <glib.h>
#include "cudart_nw_internal.h"

#include <stdio.h>

struct fatbin_wrapper {
    uint32_t magic;
    uint32_t seq;
    uint64_t ptr;
    uint64_t data_ptr;
};

struct kernel_arg {
    char is_handle;
    uint32_t size;
};

#define MAX_KERNEL_ARG 25
#define MAX_KERNEL_NAME_LEN 1024
#define MAX_ASYNC_BUFFER_NUM 16

struct fatbin_function {
    int argc;
    struct kernel_arg args[MAX_KERNEL_ARG];

    CUfunction cufunc;
    void *hostfunc;
    CUmodule module;
};
ava_end_utility;

ava_type(cudaError_t) {
    ava_success(CUDA_SUCCESS);
}

ava_type(CUresult) {
    ava_success(CUDA_SUCCESS);
}

typedef struct {
    /* argument types */
    GHashTable *fatbin_funcs;     /* for NULL, the hash table */
    int num_funcs;
    struct fatbin_function *func; /* for functions */

    /* global states */
    CUmodule cur_module;

    /* memory flags */
    int is_pinned;
} Metadata;

ava_register_metadata(Metadata);

ava_type(struct fatbin_wrapper) {
    struct fatbin_wrapper *ava_self;

    ava_field(magic);
    ava_field(seq);
    ava_field(ptr) {
        ava_type_cast(void *);
        ava_in; ava_buffer(((struct fatBinaryHeader *)ava_self->ptr)->headerSize + ((struct fatBinaryHeader *)ava_self->ptr)->fatSize);
        ava_lifetime_static;
    }
    ava_field(data_ptr) {
        ava_self->data_ptr = 0;
    }
}

ava_type(struct cudaDeviceProp);

ava_type(struct cudaPointerAttributes) {
    ava_field(devicePointer) ava_handle;
    ava_field(hostPointer) ava_opaque;
};

/* APIs needed for a minimal program */

char CUDARTAPI
__cudaInitModule(void **fatCubinHandle)
{
    ava_argument(fatCubinHandle) {
        ava_in; ava_buffer(1);
        ava_element ava_handle;
    }
}

ava_utility int __helper_cubin_num(void **cubin_handle) {
    int num = 0;
    while (cubin_handle[num] != NULL)
        num++;
    return num;
}

ava_utility void __helper_dump_fatbin(void *fatCubin,
                                    GHashTable **fatbin_funcs,
                                    int *num_funcs) {
    struct fatbin_wrapper *wp = fatCubin;
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)wp->ptr;

    /* Dump fat binary to a file */
    int fd;
    fd = open("/tmp/fatbin.cubin", O_WRONLY | O_TRUNC | O_CREAT, 0666);
    write(fd, (const void *)wp->ptr, fbh->headerSize + fbh->fatSize);
    close(fd);

    /* Execute cuobjdump and construct function information table */
    FILE *fp_pipe;
    char line[2048];
    int i, ordinal;
    size_t size;
    char name[MAX_KERNEL_NAME_LEN]; /* mangled name */
    struct fatbin_function *func;

    /* Create the hash table */
    if (*fatbin_funcs == NULL) {
        *fatbin_funcs = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
        *num_funcs = 0;
    }

    /*  Open the command pipe for reading */
    fp_pipe = popen("/usr/local/cuda-10.1/bin/cuobjdump -elf /tmp/fatbin.cubin", "r");
    assert(fp_pipe);

    while (fgets(line, sizeof(line), fp_pipe) != NULL) {
        /* Search functions */
        if (strncmp(line, ".nv.info.", 9) == 0) {
            sprintf(name, line + 9, strlen(line) - 10);
            assert(strlen(line) - 10 < MAX_KERNEL_NAME_LEN);
            name[strlen(line) - 10] = '\0';
            DEBUG_PRINT("[%d] %s@\n", *num_funcs, name);

            if (g_hash_table_lookup(*fatbin_funcs, name) != NULL)
                continue;

            /* Create a new hash table entry */
            func = (struct fatbin_function *)g_malloc(sizeof(struct fatbin_function));
            memset(func, 0, sizeof(struct fatbin_function));

            // TODO: parse function name to determine whether the
            // arguments are handles

            /* Search parameters */
            func->argc = 0;
            while (fgets(line, sizeof(line), fp_pipe) != NULL) {
                i = 0;
                while (i < strlen(line) && isspace(line[i])) i++;
                /* Empty line means reaching the end of the function info */
                if (i == strlen(line)) break;

                if (strncmp(&line[i], "Attribute:", 10) == 0) {
                    i += 10;
                    while (i < strlen(line) && isspace(line[i])) i++;
                    if (strncmp(&line[i], "EIATTR_KPARAM_INFO", 18) == 0) {
                        /* Skip the format line */
                        fgets(line, sizeof(line), fp_pipe);
                        fgets(line, sizeof(line), fp_pipe);

                        /* Get ordinal and size */
                        i = 0;
                        while (i < strlen(line) && line[i] != 'O') i++;
                        sscanf(&line[i], "Ordinal\t: 0x%x", &ordinal);
                        while (i < strlen(line) && line[i] != 'S') i++;
                        sscanf(&line[i], "Size\t: 0x%lx", &size);

                        i = func->argc;
                        //DEBUG_PRINT("ordinal=%d, size=%lx\n", ordinal, size);
                        assert(ordinal < MAX_KERNEL_ARG);
                        func->args[ordinal].size = size;
                        ++(func->argc);
                    }
                }
            }

            ++(*num_funcs);

            /* Insert the function into hash table */
            g_hash_table_insert((*fatbin_funcs), g_strdup(name), (gpointer)func);
            //func = (struct fatbin_function *)g_hash_table_lookup(*fatbin_funcs, name);
        }
    }

    pclose(fp_pipe);
}

ava_utility void __helper_print_fatcubin_info(void *fatCubin, void **ret) {
    struct fatbin_wrapper *wp = fatCubin;
    printf("fatCubin_wrapper=%p, []={.magic=0x%X, .seq=%d, ptr=0x%lx, data_ptr=0x%lx}\n",
            fatCubin,
            wp->magic, wp->seq, wp->ptr, wp->data_ptr);
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)wp->ptr;
    printf("fatBinaryHeader={.magic=0x%X, version=%d, headerSize=0x%x, fatSize=0x%llx}\n",
            fbh->magic, fbh->version, fbh->headerSize, fbh->fatSize);
    char *fatBinaryEnd = (char *)(wp->ptr + fbh->headerSize + fbh->fatSize);
    printf("fatBin=0x%lx--0x%lx\n", wp->ptr, (int64_t)fatBinaryEnd);

    fatBinaryEnd = (char *)(wp->ptr);
    int i, j;
    for (i = 0; i < 100; i++)
        if (fatBinaryEnd[i] == 0x7F && fatBinaryEnd[i+1] == 'E' && fatBinaryEnd[i+2] == 'L') {
            printf("ELF header appears at 0x%d (%p): \n", i, (void *)wp->ptr + i);
            break;
        }
    for (j = i; j < i + 32; j++)
        printf("%.2X ", fatBinaryEnd[j] & 0xFF);
    printf("\n");

    printf("ret=%p\n", ret);
    printf("fatCubin=%p, *ret=%p\n", (void *)fatCubin, *ret);
}

ava_utility void __helper_init_module(struct fatbin_wrapper *fatCubin, void **handle) {
    int ret;
    if (ava_metadata(NULL)->cur_module == 0) {
        ret = cuInit(0);
        DEBUG_PRINT("ret=%d\n", ret);
        assert(ret == CUDA_SUCCESS && "CUDA driver init failed");
    }
    __cudaInitModule(handle);
    ret = cuModuleLoadData(&ava_metadata(NULL)->cur_module, (void *)fatCubin->ptr);
    DEBUG_PRINT("ret=%d, module=%lx\n", ret, (uintptr_t)ava_metadata(NULL)->cur_module);
    assert(ret == CUDA_SUCCESS && "Module load failed");
}

void** CUDARTAPI
__cudaRegisterFatBinary(void *fatCubin)
{
    ava_argument(fatCubin) {
        ava_type_cast(struct fatbin_wrapper *);
        ava_in; ava_buffer(1);
        ava_lifetime_static;
    }

    void **ret = (void **)ava_execute();
    ava_return_value {
        ava_out; ava_buffer(__helper_cubin_num(ret) + 1);
        ava_element {
            if (ret[ava_index] != NULL) ava_handle;
        }
        ava_allocates;
        ava_lifetime_manual;
    }

    __helper_dump_fatbin(fatCubin, &ava_metadata(NULL)->fatbin_funcs,
                        &ava_metadata(NULL)->num_funcs);

    if (ava_is_worker) {
        //__helper_print_fatcubin_info(fatCubin, ret);
        __helper_init_module(fatCubin, ret);
    }
}

ava_utility void __helper_unregister_fatbin(void **fatCubinHandle) {
    // free(fatCubinHandle);
    return;
}

void CUDARTAPI
__cudaUnregisterFatBinary(void **fatCubinHandle)
{
    ava_disable_native_call;

    ava_argument(fatCubinHandle) {
        ava_in;
        /*
        ava_buffer(__helper_cubin_num(fatCubinHandle) + 1);
        ava_element {
            if (fatCubinHandle[ava_index] != NULL) ava_handle;
        }
        */
        ava_buffer(1);
        ava_element ava_handle;
        ava_deallocates;
    }

    if (ava_is_worker) {
        __helper_unregister_fatbin(fatCubinHandle);
    }
}

ava_utility void __helper_assosiate_function(GHashTable *funcs,
                                            struct fatbin_function **func,
                                            void *local,
                                            const char *deviceName) {
    if (*func != NULL) {
        DEBUG_PRINT("Function (%s) metadata (%p) already exists\n",
                deviceName, local);
        return;
    }

    *func = (struct fatbin_function *)g_hash_table_lookup(funcs, deviceName);
    assert(*func && "device function not found!");
}

ava_utility void __helper_register_function(struct fatbin_function *func,
                                            const char *hostFun,
                                            CUmodule module,
                                            const char *deviceName) {
    assert(func != NULL);
    /* Only register the first host function */
    if (func->hostfunc != NULL) return;

    CUresult ret = cuModuleGetFunction(&func->cufunc, module, deviceName);
    assert(ret == CUDA_SUCCESS);
    DEBUG_PRINT("register host func %lx -> device func %lx\n", (uintptr_t)hostFun, (uintptr_t)func->cufunc);
    func->hostfunc = (void *)hostFun;
    func->module = module;
}

ava_utility void __helper_parse_function_args(const char *name, struct kernel_arg *args)
{
    int i = 0, skip = 0;

    int argc = 0;
    if (strncmp(name, "_Z", 2)) abort();
    printf("kernel=%s\n", name);

    i = 2;
    while (i < strlen(name) && isdigit(name[i])) {
        skip = skip * 10 + name[i] - '0';
        i++;
    }

    i += skip;
    while (i < strlen(name)) {
        switch(name[i]) {
            case 'P':
                args[argc++].is_handle = 1;

                /* skip qualifiers */
                if (strchr("rVK", name[i+1]) != NULL)
                    i++;

                if (i + 1 < strlen(name) && (strchr("fijl", name[i+1]) != NULL))
                    i++;
                else if (i + 1 < strlen(name) && isdigit(name[i+1])) {
                    skip = 0;
                    while (i + 1 < strlen(name) && isdigit(name[i+1])) {
                        skip = skip * 10 + name[i+1] - '0';
                        i++;
                    }
                    i += skip;
                }
                else
                    abort();
                break;

            case 'f': /* float */
            case 'i': /* int */
            case 'j': /* unsigned int */
            case 'l': /* long */
                args[argc++].is_handle = 0;
                break;

            case 'S':
                args[argc++].is_handle = 1;
                while (i < strlen(name) && name[i] != '_') i++;
                break;

            case 'v':
                i = strlen(name);
                break;

            case 'r': /* restrict (C99) */
            case 'V': /* volatile */
            case 'K': /* const */
                break;

            default:
                abort();
        }
        i++;
    }

    for (i = 0; i < argc; i++) {
        DEBUG_PRINT("function arg#%d it is %sa handle\n", i, args[i].is_handle?"":"not ");
    }
}

void CUDARTAPI
__cudaRegisterFunction(
        void   **fatCubinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize)
{
    ava_disable_native_call;

    DEBUG_PRINT("register hostFun=%p, deviceFun=%s, deviceName=%s, thread_limit=%d, tid={%d,%d,%d}, bid={%d,%d,%d}, bDim={%d,%d,%d}, gDim={%d,%d,%d}\n",
            (void *)hostFun, deviceFun, deviceName, thread_limit,
            tid?tid->x:0, tid?tid->y:0, tid?tid->z:0,
            bid?bid->x:0, bid?bid->y:0, bid?bid->z:0,
            bDim?bDim->x:0, bDim?bDim->y:0, bDim?bDim->z:0,
            gDim?gDim->x:0, gDim?gDim->y:0, gDim?gDim->z:0);

    ava_argument(fatCubinHandle) {
        ava_in; ava_buffer(__helper_cubin_num(fatCubinHandle) + 1);
        ava_element {
            if (fatCubinHandle[ava_index] != NULL) ava_handle;
        }
    }

    ava_argument(hostFun) {
        ava_opaque;
    }

    ava_argument(deviceFun) {
        ava_in; ava_buffer(strlen(deviceFun) + 1);
    }

    ava_argument(deviceName) {
        ava_in; ava_buffer(strlen(deviceName) + 1);
    }

    __helper_assosiate_function(ava_metadata(NULL)->fatbin_funcs,
                &ava_metadata(hostFun)->func, (void *)hostFun,
                deviceName);

    ava_argument(tid) {
        ava_in; ava_buffer(1);
    }
    ava_argument(bid) {
        ava_in; ava_buffer(1);
    }
    ava_argument(bDim) {
        ava_in; ava_buffer(1);
    }
    ava_argument(gDim) {
        ava_in; ava_buffer(1);
    }
    ava_argument(wSize) {
        ava_in; ava_buffer(1);
    }

    if (ava_is_worker) {
        __helper_register_function(ava_metadata(hostFun)->func, hostFun,
                ava_metadata(NULL)->cur_module, deviceName);
    }
}

ava_begin_replacement;
void CUDARTAPI
__cudaRegisterVar(
        void **fatCubinHandle,
        char  *hostVar,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global)
{
}

void CUDARTAPI
__cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
#warning This API is called for CUDA 10.1 and 10.2, but it seems to be able to be ignored.
}
ava_end_replacement;

__host__ __device__ unsigned CUDARTAPI
__cudaPushCallConfiguration(dim3   gridDim,
                            dim3   blockDim,
                            size_t sharedMem, // CHECKME: default argument in header
                            void   *stream)
{
    ava_argument(stream) {
        ava_handle;
    }
}

cudaError_t CUDARTAPI
__cudaPopCallConfiguration(dim3   *gridDim,
                           dim3   *blockDim,
                           size_t *sharedMem,
                           void   *stream)
{
    ava_argument(gridDim) {
        ava_out; ava_buffer(1);
    }
    ava_argument(blockDim) {
        ava_out; ava_buffer(1);
    }
    ava_argument(sharedMem) {
        ava_out; ava_buffer(1);
    }
    ava_argument(stream) {
        ava_type_cast(CUstream *);
        ava_out; ava_buffer(1);
        ava_element { ava_handle; }
    }
}

ava_utility void __helper_print_kernel_info(struct fatbin_function *func, void **args) {
    DEBUG_PRINT("function metadata (%p) for local %p, cufunc %p, argc %d\n",
            (void *)func, func->hostfunc, (void *)func->cufunc, func->argc);
    int i;
    for (i = 0; i < func->argc; i++) {
        DEBUG_PRINT("arg[%d] is %sa handle, size = %u, ptr = %p, content = %p\n", i,
                func->args[i].is_handle?"":"not ",
                func->args[i].size, args[i], *((void **)args[i]));
    }
}

ava_utility cudaError_t __helper_launch_kernel(struct fatbin_function *func,
                                            const void *hostFun,
                                            dim3 gridDim,
                                            dim3 blockDim,
                                            void **args,
                                            size_t sharedMem,
                                            cudaStream_t stream) {
    cudaError_t ret = (cudaError_t)CUDA_ERROR_PROFILER_ALREADY_STOPPED;

    if (func == NULL) return (cudaError_t)CUDA_ERROR_INVALID_PTX;

    if (func->hostfunc != hostFun) {
        fprintf(stderr, "search host func %p -> stored %p (device func %p)\n",
                hostFun, (void *)func->hostfunc, (void *)func->cufunc);
    }
    else {
        DEBUG_PRINT("matched host func %p -> device func %p\n", hostFun, (void *)func->cufunc);
    }
    __helper_print_kernel_info(func, args);
    ret = (cudaError_t)cuLaunchKernel(func->cufunc, gridDim.x, gridDim.y, gridDim.z,
                         blockDim.x, blockDim.y, blockDim.z,
                         sharedMem, (CUstream)stream,
                         args, NULL);

    return ret;
}

__host__ cudaError_t CUDARTAPI
cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
        size_t sharedMem, cudaStream_t stream)
{
    ava_disable_native_call;

    ava_argument(func) {
        ava_opaque;
    }

    ava_argument(args) {
        ava_in; ava_buffer(ava_metadata(func)->func->argc);
        ava_element {
            // FIXME: use the generated index name in the spec to
            // reference the outer loop's loop index at this moment.
            if (ava_metadata(func)->func->args[__args_index_0].is_handle) {
                ava_type_cast(void *);
                ava_buffer(ava_metadata(func)->func->args[__args_index_0].size);
                //ava_element ava_handle;
            }
            else {
                ava_type_cast(void *);
                ava_buffer(ava_metadata(func)->func->args[__args_index_0].size);
            }
        }
    }

    ava_argument(stream) {
        ava_handle;
    }

    cudaError_t ret;
    if (ava_is_worker) {
        ret = __helper_launch_kernel(ava_metadata(func)->func, func,
                                    gridDim, blockDim, args, sharedMem, stream);
#warning This will bypass the resource reporting routine.
        return ret;
    }
}

ava_begin_replacement;
__host__ cudaError_t CUDARTAPI
cudaMallocHost(void **ptr, size_t size)
{
    *ptr = malloc(size);
}

__host__ cudaError_t CUDARTAPI
cudaFreeHost(void *ptr)
{
    free(ptr);
}
ava_end_replacement;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaMalloc(void **devPtr, size_t size)
{
    ava_argument(devPtr) {
        ava_out; ava_buffer(1);
        ava_element ava_opaque;
    }
}

__host__ cudaError_t CUDARTAPI
cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    ava_argument(dst) {
        if (kind == cudaMemcpyHostToDevice) {
            ava_opaque;
        }
        else if (kind == cudaMemcpyDeviceToHost) {
            ava_out; ava_buffer(count);
        }
    }

    ava_argument(src) {
        if (kind == cudaMemcpyHostToDevice) {
            ava_in; ava_buffer(count);
        }
        else if (kind == cudaMemcpyDeviceToHost) {
            ava_opaque;
        }
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaFree(void *devPtr)
{
    ava_argument(devPtr) ava_opaque;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDevice(int *device)
{
    ava_argument(device) {
        ava_out; ava_buffer(1);
    }
}

__cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDeviceCount(int *count)
{
    ava_argument(count) {
        ava_out; ava_buffer(1);
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    ava_argument(prop) {
        ava_out; ava_buffer(1);
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
    ava_argument(value) {
        ava_out; ava_buffer(1);
    }
}

__host__ cudaError_t CUDARTAPI
cudaDeviceReset(void);

__host__ cudaError_t CUDARTAPI
cudaSetDevice(int device);

__host__ cudaError_t CUDARTAPI
cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    ava_argument(symbol) {
        ava_opaque;
    }
    ava_argument(src) {
        ava_in; ava_buffer(count);
    }
}

__host__ cudaError_t CUDARTAPI
cudaMemset(void *devPtr, int value, size_t count)
{
    ava_argument(devPtr) ava_handle;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceSynchronize(void);

__host__ cudaError_t CUDARTAPI
cudaEventCreate(cudaEvent_t *event)
{
    ava_argument(event) {
        ava_out; ava_buffer(1);
        ava_element ava_handle;
    }
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    ava_argument(event) ava_handle;
    ava_argument(stream) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaEventQuery(cudaEvent_t event)
{
    ava_argument(event) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    ava_argument(ms) {
        ava_out; ava_buffer(1);
    }
    ava_argument(start) ava_handle;
    ava_argument(end) ava_handle;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaEventDestroy(cudaEvent_t event)
{
    ava_argument(event) ava_handle;
}

__host__ cudaError_t CUDARTAPI
cudaEventSynchronize(cudaEvent_t event)
{
    ava_argument(event) ava_handle;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetLastError(void);

__host__ __cudart_builtin__ const char* CUDARTAPI
cudaGetErrorString(cudaError_t error)
{
    const char *ret = ava_execute();
    ava_return_value {
        ava_out; ava_buffer(strlen(ret) + 1);
        ava_lifetime_static;
    }
}

__host__ cudaError_t CUDARTAPI
cudaMemGetInfo(size_t *_free, size_t *total)
{
    ava_argument(_free) {
        ava_out; ava_buffer(1);
    }
    ava_argument(total) {
        ava_out; ava_buffer(1);
    }
}

/* CUDA driver API */

CUresult CUDAAPI
cuInit(unsigned int Flags);

CUresult CUDAAPI
cuModuleGetFunction(CUfunction *hfunc,
                    CUmodule hmod,
                    const char *name)
{
    ava_argument(hfunc) {
        ava_out; ava_buffer(1);
    }
    ava_argument(name) {
        ava_in; ava_buffer(strlen(name) + 1);
    }

    ava_execute();
    __helper_parse_function_args(name, ava_metadata(*hfunc)->func->args);
}

ava_utility size_t __helper_fatbin_size(const void *cubin) {
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)cubin;
    return fbh->fatSize + fbh->headerSize;
}

CUresult CUDAAPI
cuModuleLoadData(CUmodule *module, const void *image)
{
    ava_argument(module) {
        ava_out; ava_buffer(1);
    }
    ava_argument(image) {
        ava_in; ava_buffer(__helper_fatbin_size(image));
    }
}

CUresult CUDAAPI
cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    ava_unsupported;
}

ava_utility size_t __helper_launch_extra_size(void **extra) {
    size_t size = 1;
    while (extra[size - 1] != CU_LAUNCH_PARAM_END)
        size++;
    return size;
}

CUresult CUDAAPI
cuLaunchKernel(CUfunction f,
               unsigned int gridDimX,
               unsigned int gridDimY,
               unsigned int gridDimZ,
               unsigned int blockDimX,
               unsigned int blockDimY,
               unsigned int blockDimZ,
               unsigned int sharedMemBytes,
               CUstream hStream,
               void **kernelParams,
               void **extra)
{
    ava_argument(hStream) ava_handle;

    ava_argument(kernelParams) {
        ava_in; ava_buffer(ava_metadata(f)->func->argc);
        ava_element {
            // FIXME: use the generated index name in the spec to
            // reference the outer loop's loop index at this moment.
            if (ava_metadata(f)->func->args[__kernelParams_index_0].is_handle) {
                ava_type_cast(void *);
                ava_buffer(ava_metadata(f)->func->args[__kernelParams_index_0].size);
                ava_element ava_handle;
            }
            else {
                ava_type_cast(void *);
                ava_buffer(ava_metadata(f)->func->args[__kernelParams_index_0].size);
            }
        }
    }

    ava_argument(extra) {
        ava_in; ava_buffer(__helper_launch_extra_size(extra));
#warning The buffer size below states that every kernelParams[i] is 1 byte long.
        ava_element ava_buffer(1);
    }
}

CUresult
cuGetExportTable(const void **ppExportTable, const CUuuid * pExportTableId)
{
    ava_unsupported;
}
