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

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <dlfcn.h>
#include <string.h>
#include <fcntl.h>

#include "hugetlbfs.h"

#include "libhugetlbfs_internal.h"

static int heap_fd;
static int zero_fd;

static void *heapbase;
static void *heaptop;
static long mapsize;
static long hpage_size;

static long hugetlbfs_next_addr(long addr)
{
#if defined(__powerpc64__)
	return ALIGN(addr, 1L << SLICE_HIGH_SHIFT);
#elif defined(__powerpc__)
	return ALIGN(addr, 1L << SLICE_LOW_SHIFT);
#elif defined(__ia64__)
	if (addr < (1UL << SLICE_HIGH_SHIFT))
		return ALIGN(addr, 1UL << SLICE_HIGH_SHIFT);
	else
		return ALIGN(addr, hpage_size);
#else
	return ALIGN(addr, hpage_size);
#endif
}

/*
 * Our plan is to ask for pages 'roughly' at the BASE.  We expect and
 * require the kernel to offer us sequential pages from wherever it
 * first gave us a page.  If it does not do so, we return the page and
 * pretend there are none this covers us for the case where another
 * map is in the way.  This is required because 'morecore' must have
 * 'sbrk' semantics, ie. return sequential, contigious memory blocks.
 * Luckily, if it does not do so and we error out malloc will happily
 * go back to small pages and use mmap to get them.  Hurrah.
 */
static void *hugetlbfs_morecore(ptrdiff_t increment)
{
	int ret;
	void *p;
	long delta;
	int mmap_reserve = __hugetlb_opts.no_reserve ? MAP_NORESERVE : 0;

	INFO("hugetlbfs_morecore(%ld) = ...\n", (long)increment);

	/*
	 * how much to grow the heap by =
	 * 	(size of heap) + malloc request - mmap'd space
	 */
	delta = (heaptop-heapbase) + increment - mapsize;

	INFO("heapbase = %p, heaptop = %p, mapsize = %lx, delta=%ld\n",
	      heapbase, heaptop, mapsize, delta);

	/* align to multiple of hugepagesize. */
	delta = ALIGN(delta, hpage_size);

	if (delta > 0) {
		/* growing the heap */

		INFO("Attempting to map %ld bytes\n", delta);

		/* map in (extend) more of the file at the end of our last map */
		p = mmap(heapbase + mapsize, delta, PROT_READ|PROT_WRITE,
			 MAP_PRIVATE|mmap_reserve, heap_fd, mapsize);
		if (p == MAP_FAILED) {
			WARNING("New heap segment map at %p failed: %s\n",
				heapbase+mapsize, strerror(errno));
			return NULL;
		}

		/* if this is the first map */
		if (! mapsize) {
			if (heapbase && (heapbase != p)) {
				WARNING("Heap originates at %p instead of %p\n",
					p, heapbase);
				if (__hugetlbfs_debug)
					dump_proc_pid_maps();
			}
			/* then setup the heap variables */
			heapbase = heaptop = p;
		} else if (p != (heapbase + mapsize)) {
			/* Couldn't get the mapping where we wanted */
			munmap(p, delta);
			WARNING("New heap segment mapped at %p instead of %p\n",
			      p, heapbase + mapsize);
			if (__hugetlbfs_debug)
				dump_proc_pid_maps();
			return NULL;
		}

		/* Fault the region to ensure accesses succeed */
		if (hugetlbfs_prefault(zero_fd, p, delta) != 0) {
			munmap(p, delta);
			return NULL;
		}

		/* we now have mmap'd further */
		mapsize += delta;
	} else if (delta < 0) {
		/* shrinking the heap */

		if (!__hugetlb_opts.shrink_ok) {
			/* shouldn't ever get here */
			WARNING("Heap shrinking is turned off\n");
			return NULL;
		}

		if (!mapsize) {
			WARNING("Can't shrink empty heap!\n");
			return NULL;
		}

		/*
		 * If we are forced to change the heapaddr from the
		 * original brk() value we have violated brk semantics
		 * (which we are not supposed to do).  This shouldn't
		 * pose a problem until glibc tries to trim the heap to an
		 * address lower than what we aligned heapaddr to.  At that
		 * point the alignment "gap" causes heap corruption.
		 * So we don't allow the heap to shrink below heapbase.
		 */
		if (mapsize + delta < 0) {  /* remember: delta is negative */
			WARNING("Unable to shrink heap below %p\n", heapbase);
			/* unmap just what is currently mapped */
			delta = -mapsize;
			/* we need heaptop + increment == heapbase, so: */
			increment = heapbase - heaptop;
		}
		INFO("Attempting to unmap %ld bytes @ %p\n", -delta,
			heapbase + mapsize + delta);
		ret = munmap(heapbase + mapsize + delta, -delta);
		if (ret) {
			WARNING("Unmapping failed while shrinking heap: "
				"%s\n", strerror(errno));
		} else {

			/*
			 * Now shrink the hugetlbfs file.
			 */
			mapsize += delta;
			ret = ftruncate(heap_fd, mapsize);
			if (ret) {
				WARNING("Could not truncate hugetlbfs file to "
					"shrink heap: %s\n", strerror(errno));
			}
		}

	}

	/* heap is continuous */
	p = heaptop;
	/* and we now have added this much more space to the heap */
	heaptop = heaptop + increment;

	INFO("... = %p\n", p);
	return p;
}

void hugetlbfs_setup_morecore(void)
{
	char *ep;
	unsigned long heapaddr;

	if (! __hugetlb_opts.morecore)
		return;
	if (strcasecmp(__hugetlb_opts.morecore, "no") == 0) {
		INFO("HUGETLB_MORECORE=%s, not setting up morecore\n",
						__hugetlb_opts.morecore);
		return;
	}

	/*
	 * Determine the page size that will be used for the heap.
	 * This can be set explicitly by setting HUGETLB_MORECORE to a valid
	 * page size string or by setting HUGETLB_DEFAULT_PAGE_SIZE.
	 */
	if (strncasecmp(__hugetlb_opts.morecore, "y", 1) == 0)
		hpage_size = gethugepagesize();
	else
		hpage_size = parse_page_size(__hugetlb_opts.morecore);

	if (hpage_size <= 0) {
		if (errno == ENOSYS)
			WARNING("Hugepages unavailable\n");
		else if (errno == EOVERFLOW)
			WARNING("Hugepage size too large\n");
		else if (errno == EINVAL)
			WARNING("Invalid huge page size\n");
		else
			WARNING("Hugepage size (%s)\n", strerror(errno));
		return;
	} else if (!hugetlbfs_find_path_for_size(hpage_size)) {
		WARNING("Hugepage size %li unavailable", hpage_size);
		return;
	}

	if (hpage_size <= 0) {
		if (errno == ENOSYS)
			WARNING("Hugepages unavailable\n");
		else if (errno == EOVERFLOW || errno == ERANGE)
			WARNING("Hugepage size too large\n");
		else
			WARNING("Hugepage size (%s)\n", strerror(errno));
		return;
	}

	heap_fd = hugetlbfs_unlinked_fd_for_size(hpage_size);
	if (heap_fd < 0) {
		WARNING("Couldn't open hugetlbfs file for morecore\n");
		return;
	}

	if (__hugetlb_opts.heapbase) {
		heapaddr = strtoul(__hugetlb_opts.heapbase, &ep, 16);
		if (*ep != '\0') {
			WARNING("Can't parse HUGETLB_MORECORE_HEAPBASE: %s\n",
			      __hugetlb_opts.heapbase);
			return;
		}
	} else {
		heapaddr = (unsigned long)sbrk(0);
		heapaddr = hugetlbfs_next_addr(heapaddr);
	}
	zero_fd = open("/dev/zero", O_RDONLY);

	INFO("setup_morecore(): heapaddr = 0x%lx\n", heapaddr);

	heaptop = heapbase = (void *)heapaddr;
	__morecore = &hugetlbfs_morecore;

	/* Set some allocator options more appropriate for hugepages */

	if (__hugetlb_opts.shrink_ok)
		mallopt(M_TRIM_THRESHOLD, hpage_size / 2);
	else
		mallopt(M_TRIM_THRESHOLD, -1);
	mallopt(M_TOP_PAD, hpage_size / 2);
	/* we always want to use our morecore, not ordinary mmap().
	 * This doesn't appear to prohibit malloc() from falling back
	 * to mmap() if we run out of hugepages. */
	mallopt(M_MMAP_MAX, 0);
}
