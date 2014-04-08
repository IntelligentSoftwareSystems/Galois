#ifndef ZLIB_UTIL_H
#define ZLIB_UTIL_H

#include <zlib.h>
class zlib_writer{
	private:
		int CHUNKSIZE;
		int UNITSIZE;
		z_stream strm;
		unsigned char *out; // output buffer
	public:
		zlib_writer(int level = Z_DEFAULT_COMPRESSION);
		~zlib_writer();
		// please set flush to Z_FINISH in the final write
		size_t write(const void *ptr, size_t size, size_t nmemb, FILE *fp, int flush = Z_NO_FLUSH);
};
int zlib_decompress(void *dest, size_t *destlen, const void *source, size_t sourcelen);

#endif
