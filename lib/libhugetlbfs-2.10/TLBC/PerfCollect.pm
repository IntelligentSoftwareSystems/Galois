#
# PerfCollect.pm
#
# This module contains all the setup, data collection, and cleanup methods
# for collecting CPU performance counter information from Ingo's perf tool.
# Licensed under LGPL 2.1 as packaged with libhugetlbfs
# (c) Eric Munson 2009

package TLBC::PerfCollect;

use warnings;
use strict;
use Carp;

use FindBin qw($Bin);
use lib "$Bin/lib";
use TLBC::DataCollect;

our @ISA = qw(TLBC::DataCollect);

my $perf_output = "/tmp/perf_" . $$ . ".data";
my $reference;
my $report;
my $perf_pid;
my $perf_bin;
my $vmlinux;
my (%map_event_name, %map_event_mask);

$map_event_name{"i386###dtlb_miss"} = "-e dTLB-miss";
$map_event_name{"x86-64###dtlb_miss"} = "-e dTLB-miss";
$map_event_name{"ppc64###dtlb_miss"} = "-e dTLB-miss";

sub _get_event()
{
	my $self = shift;
	my $arch = shift;
	my $event = shift;
	my $ret;

	$ret = $map_event_name{"$arch###$event"};
	if (not defined $ret or $ret eq "") {
		return "";
	}
	return $ret;
}

sub new()
{
	my $class = shift;
	if ($reference) {
		return $reference;
	}

	$reference = {@_};
	bless($reference, $class);
	return $reference;
}

sub setup()
{
	my $self = shift;
	$vmlinux = shift;
	my $event_name = shift;
	$perf_bin = `which perf`;
	if (!$perf_bin) {
		return 0;
	}
	chomp($perf_bin);

	my $arch = `uname -m`;
	chomp($arch);
	$arch =~ s/i.86/i386/g;
	my $event = $self->_get_event($arch, $event_name);
	if ($event eq "") {
		return 0;
	}

	my $cmd = $perf_bin . " record -a -f -o $perf_output ". $event;

	$perf_pid = fork();
	if (not defined $perf_pid) {
		return 0;
	} elsif ($perf_pid == 0) {
		exec $cmd or die "Failed to start perf monitor\n";
	} else {
		return $self;
	}
}

sub samples()
{
	return 1;
}

sub get_current_eventcount()
{
	my $self = shift;
	my $binName = shift;
	my $count = 0;
	my $total;
	my $line;
	my $hits;
	my @lines = split(/\n/, $report);

	# @lines[2] will contain the total number of samples
	$lines[2] =~ m/(\d+)/;
	$total = $1;

	if ($binName eq "vmlinux") {
		$binName = "kernel";
	}

	foreach $line (@lines) {
		if ($line =~ /$binName/) {
			chomp($line);
			$line =~ s/^\s+//;
			$line =~ s/\s+$//;
			$line =~ m/(\d+\.\d+)%/;
			# $1 should hold the percentage of hits for this
                        # $binName
			$count += int(($1 / 100) * $total);
		}
	}

	return $count;
}

sub read_eventcount()
{
	my $cmd = $perf_bin . " report -k $vmlinux -i $perf_output";
	$report = `$cmd`;
}

sub shutdown()
{
	my $self = shift;
	my $cmd = "kill $perf_pid";
	system($cmd);
	unlink $perf_output;
	return $self;
}

1;

