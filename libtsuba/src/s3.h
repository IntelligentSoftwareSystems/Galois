#ifndef _LIBTSUBA_S3_H_
#define _LIBTSUBA_S3_H_ 1

/* aws says buckets can only be 63 characters and object names can only be 1024
 * bytes */
#define BUCKET_BUF_LIM   64
#define OBJECT_BUF_LIM 1024

/* extra space for slashes and null terminators */
#define URI_LIM (BUCKET_BUF_LIM + OBJECT_BUF_LIM + 64)

int s3_uri_read(const char *uri, char *buf, char **bucket, char **object);
int s3_open(const char *bucket, const char *object);

#endif
