#
# OpCollect.pm
#
# This module contains all the setup, data collection, and cleanup methods
# for collecting CPU performance counter information from oprofile.
# Licensed under LGPL 2.1 as packaged with libhugetlbfs
# (c) Eric Munson 2009

package TLBC::OpCollect;

use warnings;
use strict;
use Carp;

use FindBin qw($Bin);
use lib "$Bin/lib";
use TLBC::DataCollect;

our @ISA = qw(TLBC::DataCollect);

my $reference;
my $report;
my (%event_map, %lowlevel);
my (%event_col_map, %event_name);

#use interface 'DataCollect';

sub _clear_oprofile()
{
	my $self = shift;
	system("opcontrol --reset > /dev/null 2>&1");
	system("opcontrol --stop > /dev/null 2>&1");
	system("opcontrol --reset > /dev/null 2>&1");
	system("opcontrol --deinit > /dev/null 2>&1");
	return $self;
}

sub _get_event()
{
	my $self = shift;
	my $event = shift;
	my $lowlevel_event;

	$lowlevel_event = `$Bin/oprofile_map_events.pl --event $event 2>/dev/null`;
	chomp($lowlevel_event);
	if ($lowlevel_event eq "" || $lowlevel_event !~ /^[A-Z0-9_]+:[0-9]+/) {
		die "Unable to find $event event for this CPU\n";
	}
	$event_map{$event} = $lowlevel_event;
	return $self;
}

sub _setup_oprofile()
{
	my $self = shift;
	my $vmlinux = shift;
	my $refEvents = shift;
	my $cmd = "$Bin/oprofile_start.sh --vmlinux=$vmlinux ";
	foreach my $event (@{$refEvents}) {
		$cmd .= " --event=$event";
		$self->_get_event($event);
	}
	$cmd .= " > /dev/null 2>&1";
	system($cmd) == 0 or return 0;
	return $self;
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
	my $vmlinux = shift;
	my $refEvents = shift;
	$self->_clear_oprofile();
	return $self->_setup_oprofile($vmlinux, $refEvents);
}

sub samples()
{
	my $self = shift;
	my $event = shift;
	my $count = 0;
	my $lowlevel;
	my @vals;
	$lowlevel = $event_map{$event};
	if (!$lowlevel) {
		die "Unable to locate count and event for $event for this CPU.\n";
	}
	@vals = split(/:/, $lowlevel);
	return $vals[1];
}

sub _get_column()
{
	my $self = shift;
	my $event = shift;
	my @results;
	my $line;
	my $col = $event_col_map{$event};

	if ($col) {
		return $col;
	}

	@results = split(/\n/, $report);
	foreach $line (@results) {
		if ($line =~ /$event.*\|/) {
			my @vals = split(/\|/, $line);
			my $size = @vals;

			for (my $i = 0; $i < $size; $i++) {
				if ($vals[$i] =~ /$event/) {
					$event_col_map{$event} = $i;
					return $i;
				}
			}
			die "Unable to find event column.\n";
		}
	}
	die "Unable to find column labels.\n";
}

sub get_current_eventcount()
{
	my @results;
	my $line;
	my $hits = 0;
	my $self = shift;
	my $binName = shift;
	my $event = shift;
	my $col = 0;

	my $lowlevel = $event_map{$event};
	if (!$lowlevel) {
		die "Unable to locate event for $event for this CPU.\n";
	}
	my @vals = split(/:/, $lowlevel);
	$event = $vals[0];
	# The event column in opreport only uses the first 12 letters of
	# the event name
	$event = substr($event, 0, 12);
	@results = split(/\n/, $report);
	$col = $self->_get_column($event);

	foreach $line (@results) {
		if ($line =~ /$binName/) {
			chomp($line);
			$line =~ s/^\s+//;
			$line =~ s/\s+$//;
			$line =~ s/\s+/ /g;
			my @vals = split(/ /, $line);
			$hits += $vals[$col * 2];
		}
	}
	return $hits;
}

sub read_eventcount()
{
	system("opcontrol --dump > /dev/null 2>&1");
	$report = `opreport`;
}

sub shutdown()
{
	my $self = shift;
	_clear_oprofile();
	return $self;
}

1;
