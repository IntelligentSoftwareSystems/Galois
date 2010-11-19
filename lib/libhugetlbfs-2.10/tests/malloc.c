/*
 * libhugetlbfs - Easy use of Linux hugepages
 * Copyright (C) 2005-2006 David Gibson & Adam Litke, IBM Corporation.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#include "hugetests.h"

static int block_sizes[] = {
	sizeof(int), 1024, 128*1024, 1024*1024, 16*1024*1024,
	32*1024*1024,
};
#define NUM_SIZES	(sizeof(block_sizes) / sizeof(block_sizes[0]))

int main(int argc, char *argv[])
{
	int i;
	char *env;
	int expect_hugepage = 0;
	char *p;

	test_init(argc, argv);

	env = getenv("HUGETLB_MORECORE");
	verbose_printf("HUGETLB_MORECORE=%s\n", env);
	if (env)
		expect_hugepage = 1;

	for (i = 0; i < NUM_SIZES; i++) {
		int size = block_sizes[i];

		p = malloc(size);
		if (! p)
			FAIL("malloc()");

		verbose_printf("malloc(%d) = %p\n", size, p);

		memset(p, 0, size);

		if (expect_hugepage && (test_addr_huge(p) != 1))
			FAIL("Address is not hugepage");
		if (!expect_hugepage && (test_addr_huge(p) == 1))
			FAIL("Address is unexpectedly huge");

		free(p);
	}

	PASS();
}
