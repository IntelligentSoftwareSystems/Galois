#ifdef __cplusplus
extern "C" {
# endif

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* Check to see if the name is formed in a way that tsuba expects */
static inline bool tsuba_is_uri(const char *uri) {
   return strncmp("s3://", uri, 5) == 0;
}

/* Download a file and map it */
int tsuba_open(const char *uri);

#ifdef __cplusplus
}
#endif
