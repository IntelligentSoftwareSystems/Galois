#include <sys/types.h>

constexpr size_t intsize()
{
	return sizeof(int);
}

int main(void)
{
	unsigned char size_like_int[intsize()];

	return sizeof(size_like_int) >= sizeof(int) ? 0 : 1;
}
