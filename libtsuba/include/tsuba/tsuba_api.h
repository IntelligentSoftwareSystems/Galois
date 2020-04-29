#ifndef GALOIS_LIBTSUBA_TSUBA_TSUBA_API_H_
#define GALOIS_LIBTSUBA_TSUBA_TSUBA_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/* These are marked nolint because it would be nice to stay C compatible */
#include <stdint.h>                                    /* NOLINT */
#define TSUBA_BLOCK_SIZE (4UL << 10) /* 4K */          /* NOLINT */
#define TSUBA_BLOCK_OFFSET_MASK (TSUBA_BLOCK_SIZE - 1) /* NOLINT */
#define TSUBA_BLOCK_MASK (~TSUBA_BLOCK_OFFSET_MASK)    /* NOLINT */

struct TsubaStatBuf {
  uint64_t size;
};

/* Setup and tear down */
int TsubaInit(void);
void TsubaFini(void);

/* Download a file and open it */
int TsubaOpen(const char* uri);

/* map a particular chunk of this file (partial download)
 * @begin and @size should be well aligned to TSUBA_BLOCK_SIZE
 * return value will be aligned to tsuba_block_size as well
 */
uint8_t* TsubaMmap(const char* filename, uint64_t begin, uint64_t size);
void TsubaMunmap(uint8_t* ptr);

/* Take whatever is in @data and put it a the file called @uri */
int TsubaStore(const char* uri, const uint8_t* data, uint64_t size);

/* read a (probably small) part of the file into a caller defined buffer */
int TsubaPeek(const char* filename, uint8_t* result_buffer, uint64_t begin,
              uint64_t size);

int TsubaStat(const char* filename, TsubaStatBuf* s_buf);

#ifdef __cplusplus
}
#endif

#endif
