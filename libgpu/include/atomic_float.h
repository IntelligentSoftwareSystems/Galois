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
