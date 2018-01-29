#pragma once

__device__ static float atomicMax(float* address, float val)
{
  int* address_as_i = (int*) address;
  int val_as_i = __float_as_int(val);
  int old_as_i = *address_as_i;
  float old = __int_as_float(old_as_i);
  while (old < val) {
    old_as_i = atomicCAS(address_as_i, old_as_i, val_as_i);
    old = __int_as_float(old_as_i);
  }
  return old;
}

__device__ static float atomicMin(float* address, float val)
{
  int* address_as_i = (int*) address;
  int val_as_i = __float_as_int(val);
  int old_as_i = *address_as_i;
  float old = __int_as_float(old_as_i);
  while (old > val) {
    old_as_i = atomicCAS(address_as_i, old_as_i, val_as_i);
    old = __int_as_float(old_as_i);
  }
  return old;
}

__device__ static float atomicTestMin(float* address, float val)
{
  return atomicMin(address, val);
}

__device__ static float atomicTestMax(float* address, float val)
{
  return atomicMax(address, val);
}

__device__ static uint32_t atomicTestAdd(uint32_t* address, uint32_t val)
{
  return (val == 0) ? *address : atomicAdd(address, val);
}

__device__ static uint32_t atomicTestMin(uint32_t* address, uint32_t val)
{
  uint32_t old_val = *address;
  return (old_val <= val) ? old_val : atomicMin(address, val);
}

__device__ static uint32_t atomicTestMax(uint32_t* address, uint32_t val)
{
  uint32_t old_val = *address;
  return (old_val >= val) ? old_val : atomicMax(address, val);
}

__device__ static uint64_t atomicTestAdd(uint64_t* address, uint64_t val)
{
  return (val == 0) ? *address : atomicAdd((unsigned long long int*)address, val);
}

__device__ static uint64_t atomicTestMin(uint64_t* address, uint64_t val)
{
  uint64_t old_val = *address;
  unsigned long long int val2 = val;
  return (old_val <= val) ? old_val : atomicMin((unsigned long long int*)address, val2);
}

__device__ static uint64_t atomicTestMax(uint64_t* address, uint64_t val)
{
  uint64_t old_val = *address;
  unsigned long long int val2 = val;
  return (old_val >= val) ? old_val : atomicMax((unsigned long long int*)address, val2);
}

