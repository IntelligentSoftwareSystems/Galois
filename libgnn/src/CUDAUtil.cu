#include <cuda.h>

#include "galois/CUDAUtilHostDecls.h"
#include "galois/GNNTypes.h"

DevicePersonality device_personality;
int gpudevice;

void SetCUDADeviceId(int gpu_id) { cudaSetDevice(gpu_id); }
