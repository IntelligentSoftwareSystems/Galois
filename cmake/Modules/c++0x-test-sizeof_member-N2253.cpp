struct foo {
	short bar;
	int baz;
};

int main(void)
{
	return (sizeof(foo::baz) == 4) ? 0 : 1;
}