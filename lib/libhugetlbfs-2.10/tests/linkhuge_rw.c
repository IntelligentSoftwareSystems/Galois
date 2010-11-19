/*
 * libhugetlbfs - Easy use of Linux hugepages
 * Copyright (C) 2005-2008 David Gibson & Adam Litke, IBM Corporation.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <elf.h>
#include <link.h>

#include "hugetests.h"

#define BLOCK_SIZE	16384
#define CONST	0xdeadbeef

#define BIG_INIT	{ \
	[0] = CONST, [17] = CONST, [BLOCK_SIZE-1] = CONST, \
}
static int small_data = 1;
static int big_data[BLOCK_SIZE] = BIG_INIT;

static int small_bss;
static int big_bss[BLOCK_SIZE];

const int small_const = CONST;
const int big_const[BLOCK_SIZE] = BIG_INIT;

static int static_func(int x)
{
	return x;
}

int global_func(int x)
{
	return x;
}

static struct test_entry {
	const char *name;
	void *data;
	int size;
	int writable, execable;
	int is_huge;
} testtab[] = {
#define ENT(name, exec)	{ #name, (void *)&name, sizeof(name), 0, exec, }
	ENT(small_data, 0),
	ENT(big_data, 0),
	ENT(small_bss, 0),
	ENT(big_bss, 0),
	ENT(small_const, 0),
	ENT(big_const, 0),

	/*
	 * XXX: Due to the way functions are defined in the powerPC 64-bit ABI,
	 * the following entries will point to a call stub in the data segment
	 * instead of to the code as one might think.  Therefore, test coverage
	 * is not quite as good as it could be for ppc64.
	 */
	ENT(static_func, 1),
	ENT(global_func, 1),
};

#define NUM_TESTS	(sizeof(testtab) / sizeof(testtab[0]))

static
int parse_elf(struct dl_phdr_info *info, size_t size, void *data)
{
	int i;
	unsigned long text_end, data_start;
	long *min_align = (long *)data;
	long actual_align;

	text_end = data_start = 0;
	for (i = 0; i < info->dlpi_phnum; i++) {
		if (info->dlpi_phdr[i].p_type != PT_LOAD)
			continue;

		if (info->dlpi_phdr[i].p_flags & PF_X)
			text_end = info->dlpi_phdr[i].p_vaddr +
					info->dlpi_phdr[i].p_memsz;
		else if (info->dlpi_phdr[i].p_flags & PF_W)
			data_start = info->dlpi_phdr[i].p_vaddr;

		if (text_end && data_start)
			break;
	}

	actual_align = (data_start - text_end) / 1024;
	if (actual_align < *min_align)
		FAIL("Binary not suitably aligned");

	return 1;
}

static void check_if_writable(struct test_entry *te)
{
	int pid, ret, status;


	pid = fork();
	if (pid < 0)
		FAIL("fork: %s", strerror(errno));
	else if (pid == 0) {
		(*(char *) te->data) = 0;
		exit (0);
	} else {
		ret = waitpid(pid, &status, 0);
		if (ret < 0)
			FAIL("waitpid(): %s", strerror(errno));
		if (WIFSIGNALED(status))
			te->writable = 0;
		else
			te->writable = 1;
	}
}

static void do_test(struct test_entry *te)
{
	int i;
	volatile int *p = te->data;

	check_if_writable(te);

	if (te->writable) {
		for (i = 0; i < (te->size / sizeof(*p)); i++)
			p[i] = CONST ^ i;

		barrier();

		for (i = 0; i < (te->size / sizeof(*p)); i++)
			if (p[i] != (CONST ^ i))
				FAIL("mismatch on %s", te->name);
	} else if (te->execable) {
		int (*pf)(int) = te->data;

		if ((*pf)(CONST) != CONST)
			FAIL("%s returns incorrect results", te->name);
	} else {
		/* Otherwise just read touch it */
		for (i = 0; i < (te->size / sizeof(*p)); i++)
			p[i];
	}

	te->is_huge = (test_addr_huge(te->data) == 1);
}

int main(int argc, char *argv[])
{
	int i;
	char *env;
	int elfmap_readonly, elfmap_writable;
	long hpage_size = gethugepagesize() / 1024;

	test_init(argc, argv);

	/* Test that the binary has been aligned enough by the linker */
	if ((argc > 1) && !strcmp("--test-alignment", argv[1]))
		dl_iterate_phdr(parse_elf, &hpage_size);

	env = getenv("HUGETLB_ELFMAP");
	verbose_printf("HUGETLB_ELFMAP=%s\n", env);

	elfmap_readonly = env && strchr(env, 'R');
	elfmap_writable = env && strchr(env, 'W');

	for (i = 0; i < NUM_TESTS; i++) {
		do_test(testtab + i);
	}

	verbose_printf("Hugepages used for:");
	for (i = 0; i < NUM_TESTS; i++)
		if (testtab[i].is_huge)
			verbose_printf(" %s", testtab[i].name);
	verbose_printf("\n");

	for (i = 0; i < NUM_TESTS; i++) {
		if (testtab[i].writable) {
			if (elfmap_writable && !testtab[i].is_huge)
				FAIL("%s is not hugepage", testtab[i].name);
			if (!elfmap_writable && testtab[i].is_huge)
				FAIL("%s is hugepage", testtab[i].name);
		} else if (!testtab[i].writable) {
			if (elfmap_readonly && !testtab[i].is_huge)
				FAIL("%s is not hugepage", testtab[i].name);
			if (!elfmap_readonly && testtab[i].is_huge)
				FAIL("%s is hugepage", testtab[i].name);
		}
	}
	PASS();
}
