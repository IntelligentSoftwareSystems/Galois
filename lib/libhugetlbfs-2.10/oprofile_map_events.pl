#!/usr/bin/perl
# This script attempts to map a high-level CPU event to the oprofile counter
# of the current CPU
# Licensed under LGPL 2.1 as packaged with libhugetlbfs
# (c) Mel Gorman 2008

use Getopt::Long;
use FindBin qw($Bin);
use lib "$Bin";

use TLBC::Report;
use strict;

my ($arch, $cputype);
my $opt_verbose;
my $opt_event;
my $opt_cycle_factor=1;
my $opt_event_factor=1;
my $p = "oprofile_map_events.pl";

my $oprofile_event;
my (%map_event_name, %map_event_mask);

# CPU events miss table
$map_event_name{"i386##dtlb_miss"} = "PAGE_WALK_TYPE:100000:0x01";
$map_event_name{"i386##p4##timer"} = "GLOBAL_POWER_EVENTS:100000:0x01";
$map_event_name{"i386##p4##dtlb_miss"} = "PAGE_WALK_TYPE:3000:0x01";
$map_event_name{"i386##p4##l2cache_miss"} = "BSQ_CACHE_REFERENCE:3000:0x300";
$map_event_name{"i386##p4-ht##timer"} = "GLOBAL_POWER_EVENTS:6000:0x01";
$map_event_name{"i386##p4-ht##dtlb_miss"} = "PAGE_WALK_TYPE:3000:0x01";
$map_event_name{"i386##p4-ht##l2cache_miss"} = "BSQ_CACHE_REFERENCE:6000:0x300";
$map_event_name{"i386##core##timer"} = "CPU_CLK_UNHALTED:6000";
$map_event_name{"i386##core##dtlb_miss"} = "DTLB_MISS:500";
$map_event_name{"i386##core_2##dtlb_miss"} = "DTLB_MISSES:500:0x01";
$map_event_name{"i386##core_2##timer"} = "CPU_CLK_UNHALTED:6000";
$map_event_name{"i386##core_2##instructions"} = "INST_RETIRED_ANY_P:6000";
$map_event_name{"x86-64##timer"} = "CPU_CLK_UNHALTED:100000";
$map_event_name{"x86-64##hammer##dtlb_miss"} = "L1_AND_L2_DTLB_MISSES:100000";
$map_event_name{"x86-64##hammer##l1cache_miss"} = "DATA_CACHE_MISSES:500";
$map_event_name{"x86-64##hammer##l2cache_miss"} = "L2_CACHE_MISS:500";
$map_event_name{"x86-64##family10##dtlb_miss"} = "L1_DTLB_AND_L2_DTLB_MISS:500";
$map_event_name{"x86-64##family10##l1cache_miss"} = "DATA_CACHE_MISSES:500";
$map_event_name{"x86-64##family10##l2cache_miss"} = "L2_CACHE_MISS:500";
$map_event_name{"x86-64##core_2##dtlb_miss"} = "DTLB_MISSES:500:0x01";
$map_event_name{"x86-64##core_2##timer"} = "CPU_CLK_UNHALTED:6000";
$map_event_name{"x86-64##core_2##instructions"} = "INST_RETIRED_ANY_P:6000";
$map_event_name{"ppc64##timer"} = "CYCLES:10000";
$map_event_name{"ppc64##dtlb_miss"} = "PM_DTLB_MISS_GRP44:100000";
$map_event_name{"ppc64##timer30"} = "PM_CYC_GRP30:10000";
$map_event_name{"ppc64##tablewalk_cycles"} = "PM_DATA_TABLEWALK_CYC_GRP30:1000";
$map_event_name{"ppc64##970MP##timer"} = "PM_CYC_GRP22:10000";
$map_event_name{"ppc64##970MP##dtlb_miss"} = "PM_DTLB_MISS_GRP22:1000";
$map_event_name{"ppc64##970MP##l1cache_ld_miss"} = "PM_LD_MISS_L1_GRP22:1000";
$map_event_name{"ppc64##970MP##l1cache_st_miss"} = "PM_ST_MISS_L1_GRP22:1000";
$map_event_name{"ppc64##970MP##timer50"} = "PM_CYC_GRP50:10000";
$map_event_name{"ppc64##970MP##l1l2cache_miss"} = "PM_DATA_FROM_MEM_GRP50:1000";
$map_event_name{"ppc64##970MP##timer30"} = "PM_CYC_GRP30:10000";
$map_event_name{"ppc64##970MP##tablewalk_cycles"} = "PM_DATA_TABLEWALK_CYC_GRP30:1000";
$map_event_name{"ppc64##power5##dtlb_miss"} = "PM_DTLB_MISS_GRP44:100000";
$map_event_name{"ppc64##power5##tablewalk_cycles"} = "PM_DATA_TABLEWALK_CYC_GRP44:1000";
$map_event_name{"ppc64##power4##dtlb_miss"} = "PM_DTLB_MISS_GRP9:1000";
$map_event_name{"ppc64##power4##tablewalk_cycles"} = "PM_DATA_TABLEWALK_CYC_GRP9:1000";
$map_event_name{"ppc64##power6##dtlb_miss"} = "PM_LSU_DERAT_MISS_GRP76:1000";
$map_event_name{"ppc64##power6##tablewalk_cycles"} = "PM_LSU_DERAT_MISS_CYC_GRP76:1000";
$map_event_name{"ppc64##power7##timer"} = "PM_RUN_CYC_GRP12:10000";
$map_event_name{"ppc64##power7##timer30"} = "PM_RUN_CYC_GRP86:10000";
$map_event_name{"ppc64##power7##dtlb_miss"} = "PM_DTLB_MISS_GRP12:1000";
$map_event_name{"ppc64##power7##tablewalk_cycles"} = "PM_DATA_TABLEWALK_CYC_GRP86:1000";

GetOptions(
	'verbose'			=>	\$opt_verbose,
	'sample-cycle-factor|c=n'	=>	\$opt_cycle_factor,
	'sample-event-factor|e=n'	=>	\$opt_event_factor,
	'event|e=s'			=>	\$opt_event,
	);
setVerbose if $opt_verbose;

if ($opt_event eq "" || $opt_event eq "default") {
	print "default\n";
	exit(0);
}

# Run --list-events to setup devices
open (SETUP, "opcontrol --list-events|") || die("Failed to exec opcontrol");
printVerbose("$p\::init list-events\n");
while (!eof(SETUP)) {
	$_ = <SETUP>;
}
close(SETUP);

# Read the arch and CPU type
open (CPUTYPE, "/proc/sys/dev/oprofile/cpu_type") ||
	open (CPUTYPE, "/dev/oprofile/cpu_type") ||
		die("Failed to open cpu_type oprofile device");
($arch, $cputype) = split(/\//, <CPUTYPE>);
close CPUTYPE;
printVerbose("$p\::arch = $arch\n");
printVerbose("$p\::cputype = $cputype\n");
printVerbose("$p\::event = $opt_event\n");

# Lookup the event for the processor
$oprofile_event = $map_event_name{"$arch##$cputype##$opt_event"};
printVerbose("$p\::lookup $arch##$cputype##$opt_event = $oprofile_event\n");
if ($oprofile_event eq "") {
	$oprofile_event = $map_event_name{"$arch##$opt_event"};
	printVerbose("$p\:: lookup $arch##$opt_event = $oprofile_event\n");
}

# If unknown, exit with failure
if ($oprofile_event eq "") {
	print "UNKNOWN_EVENT\n";
	exit(-2);
}

# Apply the sampling factor if specified
if ($opt_cycle_factor != 1 || $opt_event_factor != 1) {
	my ($event, $sample, $mask) = split(/:/, $oprofile_event);

	if ($opt_event =~ /^timer[0-9]*/) {
		$sample *= $opt_cycle_factor;
	} else {
		$sample *= $opt_event_factor;
	}
	if ($mask eq "") {
		$oprofile_event = "$event:$sample";
	} else {
		$oprofile_event = "$event:$sample:$mask";
	}
}

# Verify opcontrol agrees
open (VERIFY, "opcontrol --list-events|") || die("Failed to exec opcontrol");
my ($oprofile_event_name) = split(/:/, $oprofile_event);
printVerbose("$p\::checking $oprofile_event_name\n");
while (!eof(VERIFY)) {
	if (<VERIFY> =~ /^$oprofile_event_name:/) {
		close(VERIFY);
		print "$oprofile_event\n";
		exit(0);
	}
}
close(VERIFY);
printVerbose("$p\::opcontrol --list-events disagrees\n");
print "UNKNOWN_OPROFILE_DISPARITY\n";
exit(-3);
