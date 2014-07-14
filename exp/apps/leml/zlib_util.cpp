
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include "zlib_util.h"

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

zlib_writer::zlib_writer(int level) { 
	CHUNKSIZE = 2048;
	UNITSIZE = 16;
	out = MALLOC(unsigned char, CHUNKSIZE*UNITSIZE);
	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	if(deflateInit(&strm, level) != Z_OK) {
		deflateEnd(&strm);
		fprintf(stderr,"z_stream initial fails\n");
	}
}

zlib_writer::~zlib_writer() { if(out) free(out); }

// flush could be either Z_NO_FLUSH or Z_FINISH
size_t zlib_writer::write(const void *ptr, size_t size, size_t nmemb, FILE *fp, int flush){ 
	unsigned int have;
	int state;
	size_t byteswritten = 0;
	strm.avail_in = (uInt)(size*nmemb);
	strm.next_in = (unsigned char*)ptr;
	do {
		strm.avail_out = CHUNKSIZE * UNITSIZE;
		strm.next_out = out;
		state = deflate(&strm, flush);    /* no bad return value */
		assert(state != Z_STREAM_ERROR);  /* state not clobbered */
		have = CHUNKSIZE * UNITSIZE - strm.avail_out;
		if (fwrite(out, 1, have, fp) != have || ferror(fp)) {
			deflateEnd(&strm);
			fprintf(stderr,"Compression Error\n");
		}
		byteswritten += have;
	} while (strm.avail_out == 0);
	assert(strm.avail_in == 0);     /* all input will be used */
	return byteswritten;
}

int zlib_decompress(void *dest, size_t *destlen, const void *source, size_t sourcelen)
{
	int ret;
	const unsigned long CHUNKSIZE = 1073741824UL;
	z_stream strm;
	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	strm.avail_in = 0;
	strm.next_in = Z_NULL;
	ret = inflateInit(&strm);
	if (ret != Z_OK) {
		(void)inflateEnd(&strm);
		return ret;
	}
	unsigned char *in = (unsigned char *)source;
	unsigned char *out = (unsigned char *)dest;
	unsigned long bytesread = 0, byteswritten = 0;

	/* decompress until deflate stream ends or end of file */
	do {
		strm.avail_in = (uInt) min(CHUNKSIZE, sourcelen - bytesread);
		//finish all input
		if (strm.avail_in == 0)
			break;
		strm.next_in = in + bytesread;
		bytesread += strm.avail_in;

		/* run inflate() on input until output buffer not full */
		do {
			strm.avail_out = (uInt)CHUNKSIZE;
			strm.next_out = out + byteswritten;
			ret = inflate(&strm, Z_NO_FLUSH);
			assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
			switch (ret) {
				case Z_NEED_DICT:
					ret = Z_DATA_ERROR;     /* and fall through */
				case Z_DATA_ERROR:
				case Z_MEM_ERROR:
					(void)inflateEnd(&strm);
					return ret;
			}
			byteswritten += CHUNKSIZE - strm.avail_out;
		} while (strm.avail_out == 0);

		/* done when inflate() says it's done */
	} while (ret != Z_STREAM_END);

	if(byteswritten != *destlen)
		fprintf(stderr,"Compressed file corrupted (%ld v.s. %ld)\n", byteswritten, *destlen);
	*destlen = byteswritten;
	(void)inflateEnd(&strm);
	return 0;
}
