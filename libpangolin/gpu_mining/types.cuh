#ifndef TYPES_H_
#define TYPES_H_

typedef int IndexT;
typedef int index_type;
typedef uint8_t edge_data_type;
typedef uint8_t node_data_type;
typedef uint8_t key_type;
typedef uint8_t history_type;
typedef unsigned char SetType;
typedef unsigned long long AccType;

#define MAX_SIZE     5
#define WARP_SIZE   32
#define BLOCK_SIZE 256
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

#endif
