#pragma once
/*
   abitset.h

   Implements ApproxBitset. Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

template <typename T>
struct Base {};

template <>
struct Base<unsigned int> {
  enum {
    BITS     = 32,
    LOG_BITS = 5,
  };
};

template <>
struct Base<unsigned char> {
  enum {
    BITS     = 8,
    LOG_BITS = 3,
  };
};

template <typename T>
class ApproxBitset {
  int nbits;
  Shared<T> bitset;
  cudaTextureObject_t btx;
  static const unsigned int bits_per_base = Base<T>::BITS,
                            divby         = Base<T>::LOG_BITS,
                            modby         = (1 << Base<T>::LOG_BITS) - 1;
  T* bitarray;

public:
  int size;

  ApproxBitset() { nbits = 0; }

  ApproxBitset(size_t nbits) {
    this->nbits = nbits;
    // bits_per_base = sizeof(unsigned int) * 8;
    size = (nbits + bits_per_base - 1) / bits_per_base;

    // int mask = bits_per_base, count = 0;
    // while(!(mask & 1)) { mask >>=1; count++; }

    // divby = count;
    // modby = (1 << divby) - 1;

    // printf("%d: %d divby: %d, modby = %d\n", count, bits_per_base, divby,
    // modby);

    bitset.alloc(size);
    bitset.zero_gpu();
    bitarray = bitset.gpu_wr_ptr();

    cudaResourceDesc resDesc;

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType           = cudaResourceTypeLinear;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = Base<T>::BITS; // bits per channel

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    resDesc.res.linear.devPtr      = bitarray;
    resDesc.res.linear.sizeInBytes = size;
    check_cuda(cudaCreateTextureObject(&btx, &resDesc, &texDesc, NULL));
  }

  __device__ void set(int pos) {
    int elem = pos >> divby, bitpos = pos & modby;
    // printf("before %d %d: %x\n", pos, elem, bitarray[elem]);
    bitarray[elem] |= (1 << bitpos);
    // printf("after %d %d: %x\n", pos, elem, bitarray[elem]);
  }

  __device__ void unset(int pos) {
    int elem = pos >> divby, bitpos = pos & modby;
    bitarray[elem] &= ~(1 << bitpos);
  }

  __device__ int is_set(int pos) const {
    int elem = pos >> divby, bitpos = pos & modby;

    // printf("%d %d\n", bitarray[elem], tex1Dfetch<unsigned int>(btx, elem));

    // return bitarray[elem] & (1 << bitpos);
    // return tex1Dfetch<unsigned int>(btx, elem) & (1 << bitpos);
    if (!(tex1Dfetch<T>(btx, elem) & (1 << bitpos)))
      return bitarray[elem] & (1 << bitpos);
    else
      return 1;
  }

  void dump() {
    T* x = bitset.cpu_rd_ptr();
    for (int i = 0; i < size; i++) {
      printf("%d: %x\n", i, x[i]);
    }
  }
};

typedef ApproxBitset<unsigned int> ApproxBitsetInt;
typedef ApproxBitset<unsigned char> ApproxBitsetByte;
