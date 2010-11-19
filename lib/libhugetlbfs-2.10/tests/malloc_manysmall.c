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

#define ALLOC_SIZE	(128)
#define NUM_ALLOCS	(262144)

int main(int argc, char *argv[])
{
	int i;
	char *env;
	char *p;
	int expect_hugepage = 0;

	test_init(argc, argv);

	env = getenv("HUGETLB_MORECORE");
	verbose_printf("HUGETLB_MORECORE=%s\n", env);
	if (env)
		expect_hugepage = 1;

	for (i = 0; i < NUM_ALLOCS; i++) {
		p = malloc(ALLOC_SIZE);
		if (! p)
			FAIL("malloc()");

		if (i < 16)
			verbose_printf("p = %p\n", p);

		memset(p, 0, ALLOC_SIZE);

		if ((i % 157) == 0) {
			/* With this many allocs, testing every one
			 * takes forever */
			if (expect_hugepage && (test_addr_huge(p) != 1))
				FAIL("Address is not hugepage");
			if (!expect_hugepage && (test_addr_huge(p) == 1))
				FAIL("Address is unexpectedly huge");
		}
	}

	PASS();
}
