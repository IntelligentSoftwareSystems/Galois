/*
 * libhugetlbfs - Easy use of Linux hugepages
 * Copyright (C) 2008 Adam Litke, IBM Corporation.
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
#define _GNU_SOURCE 		/* For strchrnul */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/utsname.h>
#include "kernel-features.h"
#include "hugetlbfs.h"
#include "libhugetlbfs_privutils.h"
#include "libhugetlbfs_internal.h"
#include "libhugetlbfs_debug.h"

static struct kernel_version running_kernel_version;

/* This mask should always be 32 bits, regardless of the platform word size */
static unsigned int feature_mask;

static struct feature kernel_features[] = {
	[HUGETLB_FEATURE_PRIVATE_RESV] = {
		.name			= "private_reservations",
		.required_version	= "2.6.27-rc1",
	},
	[HUGETLB_FEATURE_SAFE_NORESERVE] = {
		.name			= "noreserve_safe",
		.required_version	= "2.6.34",
	}
};

static void debug_kernel_version(void)
{
	struct kernel_version *ver = &running_kernel_version;

	INFO("Parsed kernel version: [%u] . [%u] . [%u] ",
		ver->major, ver->minor, ver->release);
	if (ver->post)
		INFO_CONT(" [post-release: %u]\n", ver->post);
	else if (ver->pre)
		INFO_CONT(" [pre-release: %u]\n", ver->pre);
	else
		INFO_CONT("\n");
}

static int str_to_ver(const char *str, struct kernel_version *ver)
{
	int err;
	int nr_chars;
	char extra[4];

	/* Clear out version struct */
	ver->major = ver->minor = ver->release = ver->post = ver->pre = 0;

	/* The kernel always starts x.y.z */
	err = sscanf(str, "%u.%u.%u%n", &ver->major, &ver->minor, &ver->release,
			&nr_chars);
	/*
	 * The sscanf man page says that %n may or may not affect the return
	 * value so make sure it is at least 3 to cover the three kernel
	 * version variables and assume nr_chars will be correctly assigned.
	 */
	if (err < 3) {
		ERROR("Unable to determine base kernel version: %s\n",
			strerror(errno));
		return -1;
	}

	/* Advance the str by the number of characters indicated by sscanf */
	str += nr_chars;

	/* Try to match a post/stable version */
	err = sscanf(str, ".%u", &ver->post);
	if (err == 1)
		return 0;

	/* Try to match a preN/rcN version */
	err = sscanf(str, "-%3[^0-9]%u", extra, &ver->pre);
	if (err != 2 || (strcmp(extra, "pre") != 0 && strcmp(extra, "rc") != 0))
		ver->pre = 0;

	/*
	 * For now we ignore any extraversions besides pre and post versions
	 * and treat them as equal to the base version.
	 */
	return 0;
}

static int int_cmp(int a, int b)
{
	if (a < b)
		return -1;
	if (a > b)
		return 1;
	else
		return 0;
}

/*
 * Pre-release kernels have the following compare rules:
 * 	X.Y.(Z - 1) < X.Y.Z-rcN < X.Y.X
 * This order can be enforced by simply decrementing the release (for
 * comparison purposes) when there is a pre/rc modifier in effect.
 */
static int ver_cmp_release(struct kernel_version *ver)
{
	if (ver->pre)
		return ver->release - 1;
	else
		return ver->release;
}

static int ver_cmp(struct kernel_version *a, struct kernel_version *b)
{
	int ret, a_release, b_release;

	if ((ret = int_cmp(a->major, b->major)) != 0)
		return ret;

	if ((ret = int_cmp(a->minor, b->minor)) != 0)
		return ret;

	a_release = ver_cmp_release(a);
	b_release = ver_cmp_release(b);
	if ((ret = int_cmp(a_release, b_release)) != 0)
		return ret;

	if ((ret = int_cmp(a->post, b->post)) != 0)
		return ret;

	if ((ret = int_cmp(a->pre, b->pre)) != 0)
		return ret;

	/* We ignore forks (such as -mm and -mjb) */
	return 0;
}

int test_compare_kver(const char *a, const char *b)
{
	struct kernel_version ka, kb;

	if (str_to_ver(a, &ka) < 0)
		return -EINVAL;
	if (str_to_ver(b, &kb) < 0)
		return -EINVAL;
	return ver_cmp(&ka, &kb);
}

int hugetlbfs_test_feature(int feature_code)
{
	if (feature_code >= HUGETLB_FEATURE_NR) {
		ERROR("hugetlbfs_test_feature: invalid feature code\n");
		return -EINVAL;
	}
	return feature_mask & (1 << feature_code);
}

static void print_valid_features(void)
{
	int i;

	ERROR("HUGETLB_FEATURES=\"<feature>[,<feature>] ...\"\n");
	ERROR_CONT("Valid features:\n");
	for (i = 0; i < HUGETLB_FEATURE_NR; i++)
		ERROR_CONT("\t%s, no_%s\n", kernel_features[i].name,
						kernel_features[i].name);
}

static int check_features_env_valid(const char *env)
{
	const char *pos = env;
	int i;

	while (pos && *pos != '\0') {
		int match = 0;
		char *next;

		if (*pos == ',')
			pos++;
		next = strchrnul(pos, ',');
		if (strncmp(pos, "no_", 3) == 0)
			pos += 3;

		for (i = 0; i < HUGETLB_FEATURE_NR; i++) {
			char *name = kernel_features[i].name;
			if (strncmp(pos, name, next - pos) == 0) {
				match = 1;
				break;
			}
		}
		if (!match) {
			print_valid_features();
			return -1;
		}
		pos = next;
	}
	return 0;
}

void setup_features()
{
	struct utsname u;
	int i;

	if (uname(&u)) {
		ERROR("Getting kernel version failed: %s\n", strerror(errno));
		return;
	}

	str_to_ver(u.release, &running_kernel_version);
	debug_kernel_version();

	/* Check if the user has overrided any features */
	if (__hugetlb_opts.features &&
		check_features_env_valid(__hugetlb_opts.features) == -1) {
		ERROR("HUGETLB_FEATURES was invalid -- ignoring.\n");
		__hugetlb_opts.features = NULL;
	}

	for (i = 0; i < HUGETLB_FEATURE_NR; i++) {
		struct kernel_version ver;
		char *name = kernel_features[i].name;
		char *pos;

		str_to_ver(kernel_features[i].required_version, &ver);

		/* Has the user overridden feature detection? */
		if (__hugetlb_opts.features &&
			(pos = strstr(__hugetlb_opts.features, name))) {
			INFO("Overriding feature %s: ", name);
			/* If feature is preceeded by 'no_' then turn it off */
			if (((pos - 3) >= __hugetlb_opts.features) &&
				!strncmp(pos - 3, "no_", 3))
				INFO_CONT("no\n");
			else {
				INFO_CONT("yes\n");
				feature_mask |= (1UL << i);
			}
			continue;
		}

		/* Is the running kernel version newer? */
		if (ver_cmp(&running_kernel_version, &ver) >= 0) {
			INFO("Feature %s is present in this kernel\n",
				kernel_features[i].name);
			feature_mask |= (1UL << i);
		}
	}
}
