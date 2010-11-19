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

#define _LARGEFILE64_SOURCE
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "hugetlbfs.h"
#include "hugetests.h"

#define HUGETLBFS_MAGIC	0x958458f6
#define BUF_SZ 1024
#define MEMINFO_SZ 2048

int verbose_test = 1;
char *test_name;

void check_must_be_root(void)
{
	uid_t uid = getuid();
	if (uid != 0)
		CONFIG("Must be root");
}

void check_hugetlb_shm_group(void)
{
	int fd;
	ssize_t ret;
	char gid_buffer[64] = {0};
	gid_t hugetlb_shm_group;
	gid_t gid = getgid();
	uid_t uid = getuid();

	/* root is an exception */
	if (uid == 0)
		return;

	fd = open("/proc/sys/vm/hugetlb_shm_group", O_RDONLY);
	if (fd < 0)
		ERROR("Unable to open /proc/sys/vm/hugetlb_shm_group: %s",
							strerror(errno));
	ret = read(fd, &gid_buffer, sizeof(gid_buffer));
	if (ret < 0)
		ERROR("Unable to read /proc/sys/vm/hugetlb_shm_group: %s",
							strerror(errno));
	hugetlb_shm_group = atoi(gid_buffer);
	close(fd);
	if (hugetlb_shm_group != gid)
		CONFIG("Do not have permission to use SHM_HUGETLB");
}

void  __attribute__((weak)) cleanup(void)
{
}

#if 0
static void segv_handler(int signum, siginfo_t *si, void *uc)
{
	FAIL("Segmentation fault");
}
#endif

static void sigint_handler(int signum, siginfo_t *si, void *uc)
{
	cleanup();
	fprintf(stderr, "%s: %s (pid=%d)\n", test_name,
		strsignal(signum), getpid());
	exit(RC_BUG);
}

void test_init(int argc, char *argv[])
{
	int err;
	struct sigaction sa_int = {
		.sa_sigaction = sigint_handler,
	};

	test_name = argv[0];

	err = sigaction(SIGINT, &sa_int, NULL);
	if (err)
		FAIL("Can't install SIGINT handler: %s", strerror(errno));

	if (getenv("QUIET_TEST"))
		verbose_test = 0;

	verbose_printf("Starting testcase \"%s\", pid %d\n",
		       test_name, getpid());
}

#define MAPS_BUF_SZ 4096

static int read_maps(unsigned long addr, char *buf)
{
	FILE *f;
	char line[MAPS_BUF_SZ];
	char *tmp;

	f = fopen("/proc/self/maps", "r");
	if (!f) {
		ERROR("Failed to open /proc/self/maps: %s\n", strerror(errno));
		return -1;
	}

	while (1) {
		unsigned long start, end, off, ino;
		int ret;

		tmp = fgets(line, MAPS_BUF_SZ, f);
		if (!tmp)
			break;

		buf[0] = '\0';
		ret = sscanf(line, "%lx-%lx %*s %lx %*s %ld %255s",
			     &start, &end, &off, &ino,
			     buf);
		if ((ret < 4) || (ret > 5)) {
			ERROR("Couldn't parse /proc/self/maps line: %s\n",
			      line);
			fclose(f);
			return -1;
		}

		if ((start <= addr) && (addr < end)) {
			fclose(f);
			return 1;
		}
	}

	fclose(f);
	return 0;
}

/* We define this function standalone, rather than in terms of
 * hugetlbfs_test_path() so that we can use it without -lhugetlbfs for
 * testing PRELOAD */
int test_addr_huge(void *p)
{
	char name[256];
	char *dirend;
	int ret;
	struct statfs64 sb;

	ret = read_maps((unsigned long)p, name);
	if (ret < 0)
		return ret;
	if (ret == 0) {
		verbose_printf("Couldn't find address %p in /proc/self/maps\n",
			       p);
		return -1;
	}

	/* looks like a filename? */
	if (name[0] != '/')
		return 0;

	/* Truncate the filename portion */

	dirend = strrchr(name, '/');
	if (dirend && dirend > name) {
		*dirend = '\0';
	}

	ret = statfs64(name, &sb);
	if (ret)
		return -1;

	return (sb.f_type == HUGETLBFS_MAGIC);
}

ino_t get_addr_inode(void *p)
{
	char name[256];
	int ret;
	struct stat sb;

	ret = read_maps((unsigned long)p, name);
	if (ret < 0)
		return ret;
	if (ret == 0) {
		ERROR("Couldn't find address %p in /proc/self/maps\n", p);
		return -1;
	}

	/* Don't care about non-filenames */
	if (name[0] != '/')
		return 0;

	/* Truncate the filename portion */

	ret = stat(name, &sb);
	if (ret < 0) {
		/* Don't care about unlinked files */
		if (errno == ENOENT)
			return 0;
		ERROR("stat failed: %s\n", strerror(errno));
		return -1;
	}

	return sb.st_ino;
}

int remove_shmid(int shmid)
{
	if (shmid >= 0) {
		if (shmctl(shmid, IPC_RMID, NULL) != 0) {
			ERROR("shmctl(%x, IPC_RMID) failed (%s)\n",
			      shmid, strerror(errno));
			return -1;
		}
	}
	return 0;
}
