#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct snappy_file;

typedef struct snappy_file * SNAPPY_FILE;

SNAPPY_FILE snopen(const char *name, const char *mode);
size_t snwrite(SNAPPY_FILE f, void *p, size_t sz);
size_t snread(SNAPPY_FILE f, void *p, size_t sz);
int sneof(SNAPPY_FILE f);
void snclose(SNAPPY_FILE f);

#ifdef __cplusplus
}
#endif
